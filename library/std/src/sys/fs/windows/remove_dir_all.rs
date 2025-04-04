//! The Windows implementation of std::fs::remove_dir_all.
//!
//! This needs to address two issues:
//!
//! - It must not be possible to trick this into deleting files outside of
//!   the parent directory (see CVE-2022-21658).
//! - It should not fail if many threads or processes call `remove_dir_all`
//!   on the same path.
//!
//! The first is handled by using the low-level `NtOpenFile` API to open a file
//! relative to a parent directory.
//!
//! The second is trickier. Deleting a file works by setting its "disposition"
//! to delete. However, it isn't actually deleted until the file is closed.
//! During the gap between these two events, the file is in a kind of limbo
//! state where it still exists in the filesystem but anything trying to open
//! it fails with an error.
//!
//! The mitigations we use here are:
//!
//! - When attempting to open the file, we treat ERROR_DELETE_PENDING as a
//!   successful delete.
//! - If the file still hasn't been removed from the filesystem by the time we
//!   attempt to delete the parent directory, we try to wait for it to finish.
//!   We can't wait indefinitely though so after some number of spins, we give
//!   up and return an error.
//!
//! In short, we can't guarantee this will always succeed in the event of a
//! race but we do make a best effort such that it *should* do so.

use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};

use super::{AsRawHandle, DirBuff, File, FromRawHandle};
use crate::sys::c;
use crate::sys::pal::api::WinError;
use crate::thread;

// The maximum number of times to spin when waiting for deletes to complete.
const MAX_RETRIES: usize = 50;

/// A wrapper around a raw NtOpenFile call.
///
/// This isn't completely safe because `OBJECT_ATTRIBUTES` contains raw pointers.
unsafe fn nt_open_file(
    access: u32,
    object_attribute: &c::OBJECT_ATTRIBUTES,
    share: u32,
    options: u32,
) -> Result<File, WinError> {
    unsafe {
        let mut handle = ptr::null_mut();
        let mut io_status = c::IO_STATUS_BLOCK::PENDING;
        let status =
            c::NtOpenFile(&mut handle, access, object_attribute, &mut io_status, share, options);
        if c::nt_success(status) {
            Ok(File::from_raw_handle(handle))
        } else {
            // Convert an NTSTATUS to the more familiar Win32 error code (aka "DosError")
            let win_error = if status == c::STATUS_DELETE_PENDING {
                // We make a special exception for `STATUS_DELETE_PENDING` because
                // otherwise this will be mapped to `ERROR_ACCESS_DENIED` which is
                // very unhelpful because that can also mean a permission error.
                WinError::DELETE_PENDING
            } else {
                WinError::new(c::RtlNtStatusToDosError(status))
            };
            Err(win_error)
        }
    }
}

/// Open the file `path` in the directory `parent`, requesting the given `access` rights.
/// `options` will be OR'd with `FILE_OPEN_REPARSE_POINT`.
fn open_link_no_reparse(
    parent: &File,
    path: &[u16],
    access: u32,
    options: u32,
) -> Result<Option<File>, WinError> {
    // This is implemented using the lower level `NtOpenFile` function as
    // unfortunately opening a file relative to a parent is not supported by
    // win32 functions.
    //
    // See https://learn.microsoft.com/windows/win32/api/winternl/nf-winternl-ntopenfile

    // The `OBJ_DONT_REPARSE` attribute ensures that we haven't been
    // tricked into following a symlink. However, it may not be available in
    // earlier versions of Windows.
    static ATTRIBUTES: AtomicU32 = AtomicU32::new(c::OBJ_DONT_REPARSE);

    let result = unsafe {
        let mut path_str = c::UNICODE_STRING::from_ref(path);
        let mut object = c::OBJECT_ATTRIBUTES {
            ObjectName: &mut path_str,
            RootDirectory: parent.as_raw_handle(),
            Attributes: ATTRIBUTES.load(Ordering::Relaxed),
            ..c::OBJECT_ATTRIBUTES::with_length()
        };
        let share = c::FILE_SHARE_DELETE | c::FILE_SHARE_READ | c::FILE_SHARE_WRITE;
        let options = c::FILE_OPEN_REPARSE_POINT | options;
        let result = nt_open_file(access, &object, share, options);

        // Retry without OBJ_DONT_REPARSE if it's not supported.
        if matches!(result, Err(WinError::INVALID_PARAMETER))
            && ATTRIBUTES.load(Ordering::Relaxed) == c::OBJ_DONT_REPARSE
        {
            ATTRIBUTES.store(0, Ordering::Relaxed);
            object.Attributes = 0;
            nt_open_file(access, &object, share, options)
        } else {
            result
        }
    };

    // Ignore not found errors
    match result {
        Ok(f) => Ok(Some(f)),
        Err(
            WinError::FILE_NOT_FOUND
            | WinError::PATH_NOT_FOUND
            | WinError::BAD_NETPATH
            | WinError::BAD_NET_NAME
            // `DELETE_PENDING` means something else is already trying to delete it
            // so we assume that will eventually succeed.
            | WinError::DELETE_PENDING,
        ) => Ok(None),
        Err(e) => Err(e),
    }
}

fn open_dir(parent: &File, name: &[u16]) -> Result<Option<File>, WinError> {
    // Open the directory for synchronous directory listing.
    open_link_no_reparse(
        parent,
        name,
        c::SYNCHRONIZE | c::FILE_LIST_DIRECTORY,
        // "_IO_NONALERT" means that a synchronous call won't be interrupted.
        c::FILE_SYNCHRONOUS_IO_NONALERT,
    )
}

fn delete(parent: &File, name: &[u16]) -> Result<(), WinError> {
    // Note that the `delete` function consumes the opened file to ensure it's
    // dropped immediately. See module comments for why this is important.
    match open_link_no_reparse(parent, name, c::DELETE, 0) {
        Ok(Some(f)) => f.delete(),
        Ok(None) => Ok(()),
        Err(e) => Err(e),
    }
}

/// A simple retry loop that keeps running `f` while it fails with the given
/// error code or until `MAX_RETRIES` is reached.
fn retry<T: PartialEq>(
    mut f: impl FnMut() -> Result<T, WinError>,
    ignore: WinError,
) -> Result<T, WinError> {
    let mut i = MAX_RETRIES;
    loop {
        i -= 1;
        if i == 0 {
            return f();
        } else {
            let result = f();
            if result != Err(ignore) {
                return result;
            }
        }
        thread::yield_now();
    }
}

pub fn remove_dir_all_iterative(dir: File) -> Result<(), WinError> {
    let mut buffer = DirBuff::new();
    let mut dirlist = vec![dir];

    let mut restart = true;
    'outer: while let Some(dir) = dirlist.pop() {
        let more_data = dir.fill_dir_buff(&mut buffer, restart)?;
        for (name, is_directory) in buffer.iter() {
            if is_directory {
                let Some(subdir) = open_dir(&dir, &name)? else { continue };
                dirlist.push(dir);
                dirlist.push(subdir);
                continue 'outer;
            } else {
                // Attempt to delete, retrying on sharing violation errors as these
                // can often be very temporary. E.g. if something takes just a
                // bit longer than expected to release a file handle.
                retry(|| delete(&dir, &name), WinError::SHARING_VIOLATION)?;
            }
        }
        if more_data {
            dirlist.push(dir);
            restart = false;
        } else {
            // Attempt to delete, retrying on not empty errors because we may
            // need to wait some time for files to be removed from the filesystem.
            retry(|| delete(&dir, &[]), WinError::DIR_NOT_EMPTY)?;
            restart = true;
        }
    }
    Ok(())
}
