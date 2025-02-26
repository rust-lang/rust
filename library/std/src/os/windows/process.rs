//! Windows-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

#![stable(feature = "process_extensions", since = "1.2.0")]

use crate::ffi::{OsStr, c_void};
use crate::mem::MaybeUninit;
use crate::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
};
use crate::sealed::Sealed;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::{io, marker, process, ptr, sys};

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawHandle for process::Stdio {
    unsafe fn from_raw_handle(handle: RawHandle) -> process::Stdio {
        let handle = unsafe { sys::handle::Handle::from_raw_handle(handle as *mut _) };
        let io = sys::process::Stdio::Handle(handle);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedHandle> for process::Stdio {
    /// Takes ownership of a handle and returns a [`Stdio`](process::Stdio)
    /// that can attach a stream to it.
    fn from(handle: OwnedHandle) -> process::Stdio {
        let handle = sys::handle::Handle::from_inner(handle);
        let io = sys::process::Stdio::Handle(handle);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::Child {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().as_raw_handle() as *mut _
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for process::Child {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().handle().as_handle()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::Child {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw_handle() as *mut _
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<process::Child> for OwnedHandle {
    /// Takes ownership of a [`Child`](process::Child)'s process handle.
    fn from(child: process::Child) -> OwnedHandle {
        child.into_inner().into_handle().into_inner()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdin {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().as_raw_handle() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdout {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().as_raw_handle() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStderr {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().as_raw_handle() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStdin {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw_handle() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStdout {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw_handle() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStderr {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw_handle() as *mut _
    }
}

/// Creates a `ChildStdin` from the provided `OwnedHandle`.
///
/// The provided handle must be asynchronous, as reading and
/// writing from and to it is implemented using asynchronous APIs.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedHandle> for process::ChildStdin {
    fn from(handle: OwnedHandle) -> process::ChildStdin {
        let handle = sys::handle::Handle::from_inner(handle);
        let pipe = sys::pipe::AnonPipe::from_inner(handle);
        process::ChildStdin::from_inner(pipe)
    }
}

/// Creates a `ChildStdout` from the provided `OwnedHandle`.
///
/// The provided handle must be asynchronous, as reading and
/// writing from and to it is implemented using asynchronous APIs.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedHandle> for process::ChildStdout {
    fn from(handle: OwnedHandle) -> process::ChildStdout {
        let handle = sys::handle::Handle::from_inner(handle);
        let pipe = sys::pipe::AnonPipe::from_inner(handle);
        process::ChildStdout::from_inner(pipe)
    }
}

/// Creates a `ChildStderr` from the provided `OwnedHandle`.
///
/// The provided handle must be asynchronous, as reading and
/// writing from and to it is implemented using asynchronous APIs.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedHandle> for process::ChildStderr {
    fn from(handle: OwnedHandle) -> process::ChildStderr {
        let handle = sys::handle::Handle::from_inner(handle);
        let pipe = sys::pipe::AnonPipe::from_inner(handle);
        process::ChildStderr::from_inner(pipe)
    }
}

/// Windows-specific extensions to [`process::ExitStatus`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "exit_status_from", since = "1.12.0")]
pub trait ExitStatusExt: Sealed {
    /// Creates a new `ExitStatus` from the raw underlying `u32` return value of
    /// a process.
    #[stable(feature = "exit_status_from", since = "1.12.0")]
    fn from_raw(raw: u32) -> Self;
}

#[stable(feature = "exit_status_from", since = "1.12.0")]
impl ExitStatusExt for process::ExitStatus {
    fn from_raw(raw: u32) -> Self {
        process::ExitStatus::from_inner(From::from(raw))
    }
}

/// Windows-specific extensions to the [`process::Command`] builder.
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "windows_process_extensions", since = "1.16.0")]
pub trait CommandExt: Sealed {
    /// Sets the [process creation flags][1] to be passed to `CreateProcess`.
    ///
    /// These will always be ORed with `CREATE_UNICODE_ENVIRONMENT`.
    ///
    /// [1]: https://docs.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
    #[stable(feature = "windows_process_extensions", since = "1.16.0")]
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command;

    /// Sets the field `wShowWindow` of [STARTUPINFO][1] that is passed to `CreateProcess`.
    /// Allowed values are the ones listed in
    /// <https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow>
    ///
    /// [1]: <https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-startupinfow>
    #[unstable(feature = "windows_process_extensions_show_window", issue = "127544")]
    fn show_window(&mut self, cmd_show: u16) -> &mut process::Command;

    /// Forces all arguments to be wrapped in quote (`"`) characters.
    ///
    /// This is useful for passing arguments to [MSYS2/Cygwin][1] based
    /// executables: these programs will expand unquoted arguments containing
    /// wildcard characters (`?` and `*`) by searching for any file paths
    /// matching the wildcard pattern.
    ///
    /// Adding quotes has no effect when passing arguments to programs
    /// that use [msvcrt][2]. This includes programs built with both
    /// MinGW and MSVC.
    ///
    /// [1]: <https://github.com/msys2/MSYS2-packages/issues/2176>
    /// [2]: <https://msdn.microsoft.com/en-us/library/17w5ykft.aspx>
    #[unstable(feature = "windows_process_extensions_force_quotes", issue = "82227")]
    fn force_quotes(&mut self, enabled: bool) -> &mut process::Command;

    /// Append literal text to the command line without any quoting or escaping.
    ///
    /// This is useful for passing arguments to applications that don't follow
    /// the standard C run-time escaping rules, such as `cmd.exe /c`.
    ///
    /// # Batch files
    ///
    /// Note the `cmd /c` command line has slightly different escaping rules than batch files
    /// themselves. If possible, it may be better to write complex arguments to a temporary
    /// `.bat` file, with appropriate escaping, and simply run that using:
    ///
    /// ```no_run
    /// # use std::process::Command;
    /// # let temp_bat_file = "";
    /// # #[allow(unused)]
    /// let output = Command::new("cmd").args(["/c", &format!("\"{temp_bat_file}\"")]).output();
    /// ```
    ///
    /// # Example
    ///
    /// Run a batch script using both trusted and untrusted arguments.
    ///
    /// ```no_run
    /// #[cfg(windows)]
    /// // `my_script_path` is a path to known bat file.
    /// // `user_name` is an untrusted name given by the user.
    /// fn run_script(
    ///     my_script_path: &str,
    ///     user_name: &str,
    /// ) -> Result<std::process::Output, std::io::Error> {
    ///     use std::io::{Error, ErrorKind};
    ///     use std::os::windows::process::CommandExt;
    ///     use std::process::Command;
    ///
    ///     // Create the command line, making sure to quote the script path.
    ///     // This assumes the fixed arguments have been tested to work with the script we're using.
    ///     let mut cmd_args = format!(r#""{my_script_path}" "--features=[a,b,c]""#);
    ///
    ///     // Make sure the user name is safe. In particular we need to be
    ///     // cautious of ascii symbols that cmd may interpret specially.
    ///     // Here we only allow alphanumeric characters.
    ///     if !user_name.chars().all(|c| c.is_alphanumeric()) {
    ///         return Err(Error::new(ErrorKind::InvalidInput, "invalid user name"));
    ///     }
    ///
    ///     // now we have validated the user name, let's add that too.
    ///     cmd_args.push_str(" --user ");
    ///     cmd_args.push_str(user_name);
    ///
    ///     // call cmd.exe and return the output
    ///     Command::new("cmd.exe")
    ///         .arg("/c")
    ///         // surround the entire command in an extra pair of quotes, as required by cmd.exe.
    ///         .raw_arg(&format!("\"{cmd_args}\""))
    ///         .output()
    /// }
    /// ````
    #[stable(feature = "windows_process_extensions_raw_arg", since = "1.62.0")]
    fn raw_arg<S: AsRef<OsStr>>(&mut self, text_to_append_as_is: S) -> &mut process::Command;

    /// When [`process::Command`] creates pipes, request that our side is always async.
    ///
    /// By default [`process::Command`] may choose to use pipes where both ends
    /// are opened for synchronous read or write operations. By using
    /// `async_pipes(true)`, this behavior is overridden so that our side is
    /// always async.
    ///
    /// This is important because if doing async I/O a pipe or a file has to be
    /// opened for async access.
    ///
    /// The end of the pipe sent to the child process will always be synchronous
    /// regardless of this option.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(windows_process_extensions_async_pipes)]
    /// use std::os::windows::process::CommandExt;
    /// use std::process::{Command, Stdio};
    ///
    /// # let program = "";
    ///
    /// Command::new(program)
    ///     .async_pipes(true)
    ///     .stdin(Stdio::piped())
    ///     .stdout(Stdio::piped())
    ///     .stderr(Stdio::piped());
    /// ```
    #[unstable(feature = "windows_process_extensions_async_pipes", issue = "98289")]
    fn async_pipes(&mut self, always_async: bool) -> &mut process::Command;

    /// Executes the command as a child process with the given
    /// [`ProcThreadAttributeList`], returning a handle to it.
    ///
    /// This method enables the customization of attributes for the spawned
    /// child process on Windows systems.
    /// Attributes offer extended configurability for process creation,
    /// but their usage can be intricate and potentially unsafe.
    ///
    /// # Note
    ///
    /// By default, stdin, stdout, and stderr are inherited from the parent
    /// process.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(windows_process_extensions_raw_attribute)]
    /// use std::os::windows::io::AsRawHandle;
    /// use std::os::windows::process::{CommandExt, ProcThreadAttributeList};
    /// use std::process::Command;
    ///
    /// # struct ProcessDropGuard(std::process::Child);
    /// # impl Drop for ProcessDropGuard {
    /// #     fn drop(&mut self) {
    /// #         let _ = self.0.kill();
    /// #     }
    /// # }
    /// #
    /// let parent = Command::new("cmd").spawn()?;
    /// let parent_process_handle = parent.as_raw_handle();
    /// # let parent = ProcessDropGuard(parent);
    ///
    /// const PROC_THREAD_ATTRIBUTE_PARENT_PROCESS: usize = 0x00020000;
    /// let mut attribute_list = ProcThreadAttributeList::build()
    ///     .attribute(PROC_THREAD_ATTRIBUTE_PARENT_PROCESS, &parent_process_handle)
    ///     .finish()
    ///     .unwrap();
    ///
    /// let mut child = Command::new("cmd").spawn_with_attributes(&attribute_list)?;
    /// #
    /// # child.kill()?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    #[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
    fn spawn_with_attributes(
        &mut self,
        attribute_list: &ProcThreadAttributeList<'_>,
    ) -> io::Result<process::Child>;
}

#[stable(feature = "windows_process_extensions", since = "1.16.0")]
impl CommandExt for process::Command {
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command {
        self.as_inner_mut().creation_flags(flags);
        self
    }

    fn show_window(&mut self, cmd_show: u16) -> &mut process::Command {
        self.as_inner_mut().show_window(Some(cmd_show));
        self
    }

    fn force_quotes(&mut self, enabled: bool) -> &mut process::Command {
        self.as_inner_mut().force_quotes(enabled);
        self
    }

    fn raw_arg<S: AsRef<OsStr>>(&mut self, raw_text: S) -> &mut process::Command {
        self.as_inner_mut().raw_arg(raw_text.as_ref());
        self
    }

    fn async_pipes(&mut self, always_async: bool) -> &mut process::Command {
        // FIXME: This currently has an intentional no-op implementation.
        // For the time being our side of the pipes will always be async.
        // Once the ecosystem has adjusted, we may then be able to start making
        // use of synchronous pipes within the standard library.
        let _ = always_async;
        self
    }

    fn spawn_with_attributes(
        &mut self,
        attribute_list: &ProcThreadAttributeList<'_>,
    ) -> io::Result<process::Child> {
        self.as_inner_mut()
            .spawn_with_attributes(sys::process::Stdio::Inherit, true, Some(attribute_list))
            .map(process::Child::from_inner)
    }
}

#[unstable(feature = "windows_process_extensions_main_thread_handle", issue = "96723")]
pub trait ChildExt: Sealed {
    /// Extracts the main thread raw handle, without taking ownership
    #[unstable(feature = "windows_process_extensions_main_thread_handle", issue = "96723")]
    fn main_thread_handle(&self) -> BorrowedHandle<'_>;
}

#[unstable(feature = "windows_process_extensions_main_thread_handle", issue = "96723")]
impl ChildExt for process::Child {
    fn main_thread_handle(&self) -> BorrowedHandle<'_> {
        self.handle.main_thread_handle()
    }
}

/// Windows-specific extensions to [`process::ExitCode`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[unstable(feature = "windows_process_exit_code_from", issue = "111688")]
pub trait ExitCodeExt: Sealed {
    /// Creates a new `ExitCode` from the raw underlying `u32` return value of
    /// a process.
    ///
    /// The exit code should not be 259, as this conflicts with the `STILL_ACTIVE`
    /// macro returned from the `GetExitCodeProcess` function to signal that the
    /// process has yet to run to completion.
    #[unstable(feature = "windows_process_exit_code_from", issue = "111688")]
    fn from_raw(raw: u32) -> Self;
}

#[unstable(feature = "windows_process_exit_code_from", issue = "111688")]
impl ExitCodeExt for process::ExitCode {
    fn from_raw(raw: u32) -> Self {
        process::ExitCode::from_inner(From::from(raw))
    }
}

/// A wrapper around windows [`ProcThreadAttributeList`][1].
///
/// [1]: <https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-initializeprocthreadattributelist>
#[derive(Debug)]
#[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
pub struct ProcThreadAttributeList<'a> {
    attribute_list: Box<[MaybeUninit<u8>]>,
    _lifetime_marker: marker::PhantomData<&'a ()>,
}

#[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
impl<'a> ProcThreadAttributeList<'a> {
    /// Creates a new builder for constructing a [`ProcThreadAttributeList`].
    pub fn build() -> ProcThreadAttributeListBuilder<'a> {
        ProcThreadAttributeListBuilder::new()
    }

    /// Returns a pointer to the underling attribute list.
    #[doc(hidden)]
    pub fn as_ptr(&self) -> *const MaybeUninit<u8> {
        self.attribute_list.as_ptr()
    }
}

#[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
impl<'a> Drop for ProcThreadAttributeList<'a> {
    /// Deletes the attribute list.
    ///
    /// This method calls [`DeleteProcThreadAttributeList`][1] to delete the
    /// underlying attribute list.
    ///
    /// [1]: <https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-deleteprocthreadattributelist>
    fn drop(&mut self) {
        let lp_attribute_list = self.attribute_list.as_mut_ptr().cast::<c_void>();
        unsafe { sys::c::DeleteProcThreadAttributeList(lp_attribute_list) }
    }
}

/// Builder for constructing a [`ProcThreadAttributeList`].
#[derive(Clone, Debug)]
#[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
pub struct ProcThreadAttributeListBuilder<'a> {
    attributes: alloc::collections::BTreeMap<usize, ProcThreadAttributeValue>,
    _lifetime_marker: marker::PhantomData<&'a ()>,
}

#[unstable(feature = "windows_process_extensions_raw_attribute", issue = "114854")]
impl<'a> ProcThreadAttributeListBuilder<'a> {
    fn new() -> Self {
        ProcThreadAttributeListBuilder {
            attributes: alloc::collections::BTreeMap::new(),
            _lifetime_marker: marker::PhantomData,
        }
    }

    /// Sets an attribute on the attribute list.
    ///
    /// The `attribute` parameter specifies the raw attribute to be set, while
    /// the `value` parameter holds the value associated with that attribute.
    /// Please refer to the [Windows documentation][1] for a list of valid attributes.
    ///
    /// # Note
    ///
    /// The maximum number of attributes is the value of [`u32::MAX`]. If this
    /// limit is exceeded, the call to [`Self::finish`] will return an `Error`
    /// indicating that the maximum number of attributes has been exceeded.
    ///
    /// # Safety Note
    ///
    /// Remember that improper use of attributes can lead to undefined behavior
    /// or security vulnerabilities. Always consult the documentation and ensure
    /// proper attribute values are used.
    ///
    /// [1]: <https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute#parameters>
    pub fn attribute<T>(self, attribute: usize, value: &'a T) -> Self {
        unsafe {
            self.raw_attribute(
                attribute,
                ptr::addr_of!(*value).cast::<c_void>(),
                crate::mem::size_of::<T>(),
            )
        }
    }

    /// Sets a raw attribute on the attribute list.
    ///
    /// This function is useful for setting attributes with pointers or sizes
    /// that cannot be derived directly from their values.
    ///
    /// # Safety
    ///
    /// This function is marked as `unsafe` because it deals with raw pointers
    /// and sizes. It is the responsibility of the caller to ensure the value
    /// lives longer than the resulting [`ProcThreadAttributeList`] as well as
    /// the validity of the size parameter.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(windows_process_extensions_raw_attribute)]
    /// use std::ffi::c_void;
    /// use std::os::windows::process::{CommandExt, ProcThreadAttributeList};
    /// use std::os::windows::raw::HANDLE;
    /// use std::process::Command;
    ///
    /// #[repr(C)]
    /// pub struct COORD {
    ///     pub X: i16,
    ///     pub Y: i16,
    /// }
    ///
    /// extern "system" {
    ///     fn CreatePipe(
    ///         hreadpipe: *mut HANDLE,
    ///         hwritepipe: *mut HANDLE,
    ///         lppipeattributes: *const c_void,
    ///         nsize: u32,
    ///     ) -> i32;
    ///     fn CreatePseudoConsole(
    ///         size: COORD,
    ///         hinput: HANDLE,
    ///         houtput: HANDLE,
    ///         dwflags: u32,
    ///         phpc: *mut isize,
    ///     ) -> i32;
    ///     fn CloseHandle(hobject: HANDLE) -> i32;
    /// }
    ///
    /// let [mut input_read_side, mut output_write_side, mut output_read_side, mut input_write_side] =
    ///     [unsafe { std::mem::zeroed::<HANDLE>() }; 4];
    ///
    /// unsafe {
    ///     CreatePipe(&mut input_read_side, &mut input_write_side, std::ptr::null(), 0);
    ///     CreatePipe(&mut output_read_side, &mut output_write_side, std::ptr::null(), 0);
    /// }
    ///
    /// let size = COORD { X: 60, Y: 40 };
    /// let mut h_pc = unsafe { std::mem::zeroed() };
    /// unsafe { CreatePseudoConsole(size, input_read_side, output_write_side, 0, &mut h_pc) };
    ///
    /// unsafe { CloseHandle(input_read_side) };
    /// unsafe { CloseHandle(output_write_side) };
    ///
    /// const PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: usize = 131094;
    ///
    /// let attribute_list = unsafe {
    ///     ProcThreadAttributeList::build()
    ///         .raw_attribute(
    ///             PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
    ///             h_pc as *const c_void,
    ///             std::mem::size_of::<isize>(),
    ///         )
    ///         .finish()?
    /// };
    ///
    /// let mut child = Command::new("cmd").spawn_with_attributes(&attribute_list)?;
    /// #
    /// # child.kill()?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub unsafe fn raw_attribute<T>(
        mut self,
        attribute: usize,
        value_ptr: *const T,
        value_size: usize,
    ) -> Self {
        self.attributes.insert(
            attribute,
            ProcThreadAttributeValue { ptr: value_ptr.cast::<c_void>(), size: value_size },
        );
        self
    }

    /// Finalizes the construction of the `ProcThreadAttributeList`.
    ///
    /// # Errors
    ///
    /// Returns an error if the maximum number of attributes is exceeded
    /// or if there is an I/O error during initialization.
    pub fn finish(&self) -> io::Result<ProcThreadAttributeList<'a>> {
        // To initialize our ProcThreadAttributeList, we need to determine
        // how many bytes to allocate for it. The Windows API simplifies this
        // process by allowing us to call `InitializeProcThreadAttributeList`
        // with a null pointer to retrieve the required size.
        let mut required_size = 0;
        let Ok(attribute_count) = self.attributes.len().try_into() else {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "maximum number of ProcThreadAttributes exceeded",
            ));
        };
        unsafe {
            sys::c::InitializeProcThreadAttributeList(
                ptr::null_mut(),
                attribute_count,
                0,
                &mut required_size,
            )
        };

        let mut attribute_list = vec![MaybeUninit::uninit(); required_size].into_boxed_slice();

        // Once we've allocated the necessary memory, it's safe to invoke
        // `InitializeProcThreadAttributeList` to properly initialize the list.
        sys::cvt(unsafe {
            sys::c::InitializeProcThreadAttributeList(
                attribute_list.as_mut_ptr().cast::<c_void>(),
                attribute_count,
                0,
                &mut required_size,
            )
        })?;

        // # Add our attributes to the buffer.
        // It's theoretically possible for the attribute count to exceed a u32
        // value. Therefore, we ensure that we don't add more attributes than
        // the buffer was initialized for.
        for (&attribute, value) in self.attributes.iter().take(attribute_count as usize) {
            sys::cvt(unsafe {
                sys::c::UpdateProcThreadAttribute(
                    attribute_list.as_mut_ptr().cast::<c_void>(),
                    0,
                    attribute,
                    value.ptr,
                    value.size,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            })?;
        }

        Ok(ProcThreadAttributeList { attribute_list, _lifetime_marker: marker::PhantomData })
    }
}

/// Wrapper around the value data to be used as a Process Thread Attribute.
#[derive(Clone, Debug)]
struct ProcThreadAttributeValue {
    ptr: *const c_void,
    size: usize,
}
