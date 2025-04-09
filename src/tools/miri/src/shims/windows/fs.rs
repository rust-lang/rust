use std::fs::{Metadata, OpenOptions};
use std::io;
use std::path::PathBuf;
use std::time::SystemTime;

use bitflags::bitflags;

use crate::shims::files::{FileDescription, FileHandle};
use crate::shims::windows::handle::{EvalContextExt as _, Handle};
use crate::*;

#[derive(Debug)]
pub struct DirHandle {
    pub(crate) path: PathBuf,
}

impl FileDescription for DirHandle {
    fn name(&self) -> &'static str {
        "directory"
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(self.path.metadata())
    }

    fn close<'tcx>(
        self,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        interp_ok(Ok(()))
    }
}

/// Windows supports handles without any read/write/delete permissions - these handles can get
/// metadata, but little else. We represent that by storing the metadata from the time the handle
/// was opened.
#[derive(Debug)]
pub struct MetadataHandle {
    pub(crate) meta: Metadata,
}

impl FileDescription for MetadataHandle {
    fn name(&self) -> &'static str {
        "metadata-only"
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(Ok(self.meta.clone()))
    }

    fn close<'tcx>(
        self,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        interp_ok(Ok(()))
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CreationDisposition {
    CreateAlways,
    CreateNew,
    OpenAlways,
    OpenExisting,
    TruncateExisting,
}

impl CreationDisposition {
    fn new<'tcx>(
        value: u32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, CreationDisposition> {
        let create_always = ecx.eval_windows_u32("c", "CREATE_ALWAYS");
        let create_new = ecx.eval_windows_u32("c", "CREATE_NEW");
        let open_always = ecx.eval_windows_u32("c", "OPEN_ALWAYS");
        let open_existing = ecx.eval_windows_u32("c", "OPEN_EXISTING");
        let truncate_existing = ecx.eval_windows_u32("c", "TRUNCATE_EXISTING");

        let out = if value == create_always {
            CreationDisposition::CreateAlways
        } else if value == create_new {
            CreationDisposition::CreateNew
        } else if value == open_always {
            CreationDisposition::OpenAlways
        } else if value == open_existing {
            CreationDisposition::OpenExisting
        } else if value == truncate_existing {
            CreationDisposition::TruncateExisting
        } else {
            throw_unsup_format!("CreateFileW: Unsupported creation disposition: {value}");
        };
        interp_ok(out)
    }
}

bitflags! {
    #[derive(PartialEq)]
    struct FileAttributes: u32 {
        const ZERO = 0;
        const NORMAL = 1 << 0;
        /// This must be passed to allow getting directory handles. If not passed, we error on trying
        /// to open directories
        const BACKUP_SEMANTICS = 1 << 1;
        /// Open a reparse point as a regular file - this is basically similar to 'readlink' in Unix
        /// terminology. A reparse point is a file with custom logic when navigated to, of which
        /// a symlink is one specific example.
        const OPEN_REPARSE = 1 << 2;
    }
}

impl FileAttributes {
    fn new<'tcx>(
        mut value: u32,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, FileAttributes> {
        let file_attribute_normal = ecx.eval_windows_u32("c", "FILE_ATTRIBUTE_NORMAL");
        let file_flag_backup_semantics = ecx.eval_windows_u32("c", "FILE_FLAG_BACKUP_SEMANTICS");
        let file_flag_open_reparse_point =
            ecx.eval_windows_u32("c", "FILE_FLAG_OPEN_REPARSE_POINT");

        let mut out = FileAttributes::ZERO;
        if value & file_flag_backup_semantics != 0 {
            value &= !file_flag_backup_semantics;
            out |= FileAttributes::BACKUP_SEMANTICS;
        }
        if value & file_flag_open_reparse_point != 0 {
            value &= !file_flag_open_reparse_point;
            out |= FileAttributes::OPEN_REPARSE;
        }
        if value & file_attribute_normal != 0 {
            value &= !file_attribute_normal;
            out |= FileAttributes::NORMAL;
        }

        if value != 0 {
            throw_unsup_format!("CreateFileW: Unsupported flags_and_attributes: {value}");
        }

        if out == FileAttributes::ZERO {
            // NORMAL is equivalent to 0. Avoid needing to check both cases by unifying the two.
            out = FileAttributes::NORMAL;
        }
        interp_ok(out)
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn CreateFileW(
        &mut self,
        file_name: &OpTy<'tcx>,            // LPCWSTR
        desired_access: &OpTy<'tcx>,       // DWORD
        share_mode: &OpTy<'tcx>,           // DWORD
        security_attributes: &OpTy<'tcx>,  // LPSECURITY_ATTRIBUTES
        creation_disposition: &OpTy<'tcx>, // DWORD
        flags_and_attributes: &OpTy<'tcx>, // DWORD
        template_file: &OpTy<'tcx>,        // HANDLE
    ) -> InterpResult<'tcx, Handle> {
        // ^ Returns HANDLE
        use CreationDisposition::*;

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "CreateFileW");
        this.check_no_isolation("`CreateFileW`")?;

        // This function appears to always set the error to 0. This is important for some flag
        // combinations, which may set error code on success.
        this.set_last_error(IoError::Raw(Scalar::from_i32(0)))?;

        let file_name = this.read_path_from_wide_str(this.read_pointer(file_name)?)?;
        let mut desired_access = this.read_scalar(desired_access)?.to_u32()?;
        let share_mode = this.read_scalar(share_mode)?.to_u32()?;
        let security_attributes = this.read_pointer(security_attributes)?;
        let creation_disposition = this.read_scalar(creation_disposition)?.to_u32()?;
        let flags_and_attributes = this.read_scalar(flags_and_attributes)?.to_u32()?;
        let template_file = this.read_target_usize(template_file)?;

        let generic_read = this.eval_windows_u32("c", "GENERIC_READ");
        let generic_write = this.eval_windows_u32("c", "GENERIC_WRITE");

        let file_share_delete = this.eval_windows_u32("c", "FILE_SHARE_DELETE");
        let file_share_read = this.eval_windows_u32("c", "FILE_SHARE_READ");
        let file_share_write = this.eval_windows_u32("c", "FILE_SHARE_WRITE");

        let creation_disposition = CreationDisposition::new(creation_disposition, this)?;
        let attributes = FileAttributes::new(flags_and_attributes, this)?;

        if share_mode != (file_share_delete | file_share_read | file_share_write) {
            throw_unsup_format!("CreateFileW: Unsupported share mode: {share_mode}");
        }
        if !this.ptr_is_null(security_attributes)? {
            throw_unsup_format!("CreateFileW: Security attributes are not supported");
        }

        if attributes.contains(FileAttributes::OPEN_REPARSE) && creation_disposition == CreateAlways
        {
            throw_machine_stop!(TerminationInfo::Abort("Invalid CreateFileW argument combination: FILE_FLAG_OPEN_REPARSE_POINT with CREATE_ALWAYS".to_string()));
        }

        if template_file != 0 {
            throw_unsup_format!("CreateFileW: Template files are not supported");
        }

        // We need to know if the file is a directory to correctly open directory handles.
        // This is racy, but currently the stdlib doesn't appear to offer a better solution.
        let is_dir = file_name.is_dir();

        // BACKUP_SEMANTICS is how Windows calls the act of opening a directory handle.
        if !attributes.contains(FileAttributes::BACKUP_SEMANTICS) && is_dir {
            this.set_last_error(IoError::WindowsError("ERROR_ACCESS_DENIED"))?;
            return interp_ok(Handle::Invalid);
        }

        let desired_read = desired_access & generic_read != 0;
        let desired_write = desired_access & generic_write != 0;

        let mut options = OpenOptions::new();
        if desired_read {
            desired_access &= !generic_read;
            options.read(true);
        }
        if desired_write {
            desired_access &= !generic_write;
            options.write(true);
        }

        if desired_access != 0 {
            throw_unsup_format!(
                "CreateFileW: Unsupported bits set for access mode: {desired_access:#x}"
            );
        }

        // Per the documentation:
        // If the specified file exists and is writable, the function truncates the file,
        // the function succeeds, and last-error code is set to ERROR_ALREADY_EXISTS.
        // If the specified file does not exist and is a valid path, a new file is created,
        // the function succeeds, and the last-error code is set to zero.
        // https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
        //
        // This is racy, but there doesn't appear to be an std API that both succeeds if a
        // file exists but tells us it isn't new. Either we accept racing one way or another,
        // or we use an iffy heuristic like file creation time. This implementation prefers
        // to fail in the direction of erroring more often.
        if let CreateAlways | OpenAlways = creation_disposition
            && file_name.exists()
        {
            this.set_last_error(IoError::WindowsError("ERROR_ALREADY_EXISTS"))?;
        }

        let handle = if is_dir {
            // Open this as a directory.
            let fd_num = this.machine.fds.insert_new(DirHandle { path: file_name });
            Ok(Handle::File(fd_num))
        } else if creation_disposition == OpenExisting && !(desired_read || desired_write) {
            // Windows supports handles with no permissions. These allow things such as reading
            // metadata, but not file content.
            file_name.metadata().map(|meta| {
                let fd_num = this.machine.fds.insert_new(MetadataHandle { meta });
                Handle::File(fd_num)
            })
        } else {
            // Open this as a standard file.
            match creation_disposition {
                CreateAlways | OpenAlways => {
                    options.create(true);
                    if creation_disposition == CreateAlways {
                        options.truncate(true);
                    }
                }
                CreateNew => {
                    options.create_new(true);
                    // Per `create_new` documentation:
                    // The file must be opened with write or append access in order to create a new file.
                    // https://doc.rust-lang.org/std/fs/struct.OpenOptions.html#method.create_new
                    if !desired_write {
                        options.append(true);
                    }
                }
                OpenExisting => {} // Default options
                TruncateExisting => {
                    options.truncate(true);
                }
            }

            options.open(file_name).map(|file| {
                let fd_num =
                    this.machine.fds.insert_new(FileHandle { file, writable: desired_write });
                Handle::File(fd_num)
            })
        };

        match handle {
            Ok(handle) => interp_ok(handle),
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(Handle::Invalid)
            }
        }
    }

    fn GetFileInformationByHandle(
        &mut self,
        file: &OpTy<'tcx>,             // HANDLE
        file_information: &OpTy<'tcx>, // LPBY_HANDLE_FILE_INFORMATION
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns BOOL (i32 on Windows)
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetFileInformationByHandle");
        this.check_no_isolation("`GetFileInformationByHandle`")?;

        let file = this.read_handle(file, "GetFileInformationByHandle")?;
        let file_information = this.deref_pointer_as(
            file_information,
            this.windows_ty_layout("BY_HANDLE_FILE_INFORMATION"),
        )?;

        let fd_num = if let Handle::File(fd_num) = file {
            fd_num
        } else {
            this.invalid_handle("GetFileInformationByHandle")?
        };

        let Some(desc) = this.machine.fds.get(fd_num) else {
            this.invalid_handle("GetFileInformationByHandle")?
        };

        let metadata = match desc.metadata()? {
            Ok(meta) => meta,
            Err(e) => {
                this.set_last_error(e)?;
                return interp_ok(this.eval_windows("c", "FALSE"));
            }
        };

        let size = metadata.len();

        let file_type = metadata.file_type();
        let attributes = if file_type.is_dir() {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_DIRECTORY")
        } else if file_type.is_file() {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_NORMAL")
        } else {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_DEVICE")
        };

        // Per the Windows documentation:
        // "If the underlying file system does not support the [...] time, this member is zero (0)."
        // https://learn.microsoft.com/en-us/windows/win32/api/fileapi/ns-fileapi-by_handle_file_information
        let created = extract_windows_epoch(this, metadata.created())?.unwrap_or((0, 0));
        let accessed = extract_windows_epoch(this, metadata.accessed())?.unwrap_or((0, 0));
        let written = extract_windows_epoch(this, metadata.modified())?.unwrap_or((0, 0));

        this.write_int_fields_named(&[("dwFileAttributes", attributes.into())], &file_information)?;
        write_filetime_field(this, &file_information, "ftCreationTime", created)?;
        write_filetime_field(this, &file_information, "ftLastAccessTime", accessed)?;
        write_filetime_field(this, &file_information, "ftLastWriteTime", written)?;
        this.write_int_fields_named(
            &[
                ("dwVolumeSerialNumber", 0),
                ("nFileSizeHigh", (size >> 32).into()),
                ("nFileSizeLow", (size & 0xFFFFFFFF).into()),
                ("nNumberOfLinks", 1),
                ("nFileIndexHigh", 0),
                ("nFileIndexLow", 0),
            ],
            &file_information,
        )?;

        interp_ok(this.eval_windows("c", "TRUE"))
    }
}

/// Windows FILETIME is measured in 100-nanosecs since 1601
fn extract_windows_epoch<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    time: io::Result<SystemTime>,
) -> InterpResult<'tcx, Option<(u32, u32)>> {
    match time.ok() {
        Some(time) => {
            let duration = ecx.system_time_since_windows_epoch(&time)?;
            let duration_ticks = ecx.windows_ticks_for(duration)?;
            #[allow(clippy::cast_possible_truncation)]
            interp_ok(Some((duration_ticks as u32, (duration_ticks >> 32) as u32)))
        }
        None => interp_ok(None),
    }
}

fn write_filetime_field<'tcx>(
    cx: &mut MiriInterpCx<'tcx>,
    val: &MPlaceTy<'tcx>,
    name: &str,
    (low, high): (u32, u32),
) -> InterpResult<'tcx> {
    cx.write_int_fields_named(
        &[("dwLowDateTime", low.into()), ("dwHighDateTime", high.into())],
        &cx.project_field_named(val, name)?,
    )
}
