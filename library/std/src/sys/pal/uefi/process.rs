use r_efi::protocols::simple_text_output;

use super::helpers;
pub use crate::ffi::OsString as EnvKey;
use crate::ffi::{OsStr, OsString};
use crate::num::{NonZero, NonZeroI32};
use crate::path::Path;
use crate::sys::fs::File;
use crate::sys::pipe::AnonPipe;
use crate::sys::unsupported;
use crate::sys_common::process::{CommandEnv, CommandEnvs};
use crate::{fmt, io};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Command {
    prog: OsString,
    args: Vec<OsString>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

// passed back to std::process with the pipes connected to the child, if any
// were requested
pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

#[derive(Copy, Clone, Debug)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command { prog: program.to_os_string(), args: Vec::new(), stdout: None, stderr: None }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_os_string());
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv {
        panic!("unsupported")
    }

    pub fn cwd(&mut self, _dir: &OsStr) {
        panic!("unsupported")
    }

    pub fn stdin(&mut self, _stdin: Stdio) {
        panic!("unsupported")
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout);
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr);
    }

    pub fn get_program(&self) -> &OsStr {
        self.prog.as_ref()
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        CommandArgs { iter: self.args.iter() }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        panic!("unsupported")
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        None
    }

    pub fn spawn(
        &mut self,
        _default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        unsupported()
    }

    fn create_pipe(
        s: Stdio,
    ) -> io::Result<Option<helpers::OwnedProtocol<uefi_command_internal::PipeProtocol>>> {
        match s {
            Stdio::MakePipe => unsafe {
                helpers::OwnedProtocol::create(
                    uefi_command_internal::PipeProtocol::new(),
                    simple_text_output::PROTOCOL_GUID,
                )
            }
            .map(Some),
            Stdio::Null => unsafe {
                helpers::OwnedProtocol::create(
                    uefi_command_internal::PipeProtocol::null(),
                    simple_text_output::PROTOCOL_GUID,
                )
            }
            .map(Some),
            Stdio::Inherit => Ok(None),
        }
    }

    pub fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        let mut cmd = uefi_command_internal::Image::load_image(&self.prog)?;

        // UEFI adds the bin name by default
        if !self.args.is_empty() {
            let args = uefi_command_internal::create_args(&self.prog, &self.args);
            cmd.set_args(args);
        }

        // Setup Stdout
        let stdout = self.stdout.unwrap_or(Stdio::MakePipe);
        let stdout = Self::create_pipe(stdout)?;
        if let Some(con) = stdout {
            cmd.stdout_init(con)
        } else {
            cmd.stdout_inherit()
        };

        // Setup Stderr
        let stderr = self.stderr.unwrap_or(Stdio::MakePipe);
        let stderr = Self::create_pipe(stderr)?;
        if let Some(con) = stderr {
            cmd.stderr_init(con)
        } else {
            cmd.stderr_inherit()
        };

        let stat = cmd.start_image()?;

        let stdout = cmd.stdout()?;
        let stderr = cmd.stderr()?;

        Ok((ExitStatus(stat), stdout, stderr))
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        pipe.diverge()
    }
}

impl From<io::Stdout> for Stdio {
    fn from(_: io::Stdout) -> Stdio {
        // FIXME: This is wrong.
        // Instead, the Stdio we have here should be a unit struct.
        panic!("unsupported")
    }
}

impl From<io::Stderr> for Stdio {
    fn from(_: io::Stderr) -> Stdio {
        // FIXME: This is wrong.
        // Instead, the Stdio we have here should be a unit struct.
        panic!("unsupported")
    }
}

impl From<File> for Stdio {
    fn from(_file: File) -> Stdio {
        // FIXME: This is wrong.
        // Instead, the Stdio we have here should be a unit struct.
        panic!("unsupported")
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ExitStatus(r_efi::efi::Status);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        if self.0 == r_efi::efi::Status::SUCCESS { Ok(()) } else { Err(ExitStatusError(self.0)) }
    }

    pub fn code(&self) -> Option<i32> {
        Some(self.0.as_usize() as i32)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_str = super::os::error_string(self.0.as_usize());
        write!(f, "{}", err_str)
    }
}

impl Default for ExitStatus {
    fn default() -> Self {
        ExitStatus(r_efi::efi::Status::SUCCESS)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ExitStatusError(r_efi::efi::Status);

impl fmt::Debug for ExitStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_str = super::os::error_string(self.0.as_usize());
        write!(f, "{}", err_str)
    }
}

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0)
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> {
        NonZeroI32::new(self.0.as_usize() as i32)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(bool);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(false);
    pub const FAILURE: ExitCode = ExitCode(true);

    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

impl From<u8> for ExitCode {
    fn from(code: u8) -> Self {
        match code {
            0 => Self::SUCCESS,
            1..=255 => Self::FAILURE,
        }
    }
}

pub struct Process(!);

impl Process {
    pub fn id(&self) -> u32 {
        self.0
    }

    pub fn kill(&mut self) -> io::Result<()> {
        self.0
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        self.0
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.0
    }
}

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;

    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|x| x.as_ref())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}

#[allow(dead_code)]
mod uefi_command_internal {
    use r_efi::protocols::{loaded_image, simple_text_output};

    use super::super::helpers;
    use crate::ffi::{OsStr, OsString};
    use crate::io::{self, const_error};
    use crate::mem::MaybeUninit;
    use crate::os::uefi::env::{boot_services, image_handle, system_table};
    use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
    use crate::ptr::NonNull;
    use crate::slice;
    use crate::sys::pal::uefi::helpers::OwnedTable;
    use crate::sys_common::wstr::WStrUnits;

    pub struct Image {
        handle: NonNull<crate::ffi::c_void>,
        stdout: Option<helpers::OwnedProtocol<PipeProtocol>>,
        stderr: Option<helpers::OwnedProtocol<PipeProtocol>>,
        st: OwnedTable<r_efi::efi::SystemTable>,
        args: Option<(*mut u16, usize)>,
    }

    impl Image {
        pub fn load_image(p: &OsStr) -> io::Result<Self> {
            let path = helpers::OwnedDevicePath::from_text(p)?;
            let boot_services: NonNull<r_efi::efi::BootServices> = boot_services()
                .ok_or_else(|| const_error!(io::ErrorKind::NotFound, "Boot Services not found"))?
                .cast();
            let mut child_handle: MaybeUninit<r_efi::efi::Handle> = MaybeUninit::uninit();
            let image_handle = image_handle();

            let r = unsafe {
                ((*boot_services.as_ptr()).load_image)(
                    r_efi::efi::Boolean::FALSE,
                    image_handle.as_ptr(),
                    path.as_ptr(),
                    crate::ptr::null_mut(),
                    0,
                    child_handle.as_mut_ptr(),
                )
            };

            if r.is_error() {
                Err(io::Error::from_raw_os_error(r.as_usize()))
            } else {
                let child_handle = unsafe { child_handle.assume_init() };
                let child_handle = NonNull::new(child_handle).unwrap();

                let loaded_image: NonNull<loaded_image::Protocol> =
                    helpers::open_protocol(child_handle, loaded_image::PROTOCOL_GUID).unwrap();
                let st = OwnedTable::from_table(unsafe { (*loaded_image.as_ptr()).system_table });

                Ok(Self { handle: child_handle, stdout: None, stderr: None, st, args: None })
            }
        }

        pub fn start_image(&mut self) -> io::Result<r_efi::efi::Status> {
            self.update_st_crc32()?;

            // Use our system table instead of the default one
            let loaded_image: NonNull<loaded_image::Protocol> =
                helpers::open_protocol(self.handle, loaded_image::PROTOCOL_GUID).unwrap();
            unsafe {
                (*loaded_image.as_ptr()).system_table = self.st.as_mut_ptr();
            }

            let boot_services: NonNull<r_efi::efi::BootServices> = boot_services()
                .ok_or_else(|| const_error!(io::ErrorKind::NotFound, "Boot Services not found"))?
                .cast();
            let mut exit_data_size: usize = 0;
            let mut exit_data: MaybeUninit<*mut u16> = MaybeUninit::uninit();

            let r = unsafe {
                ((*boot_services.as_ptr()).start_image)(
                    self.handle.as_ptr(),
                    &mut exit_data_size,
                    exit_data.as_mut_ptr(),
                )
            };

            // Drop exitdata
            if exit_data_size != 0 {
                unsafe {
                    let exit_data = exit_data.assume_init();
                    ((*boot_services.as_ptr()).free_pool)(exit_data as *mut crate::ffi::c_void);
                }
            }

            Ok(r)
        }

        fn set_stdout(
            &mut self,
            handle: r_efi::efi::Handle,
            protocol: *mut simple_text_output::Protocol,
        ) {
            unsafe {
                (*self.st.as_mut_ptr()).console_out_handle = handle;
                (*self.st.as_mut_ptr()).con_out = protocol;
            }
        }

        fn set_stderr(
            &mut self,
            handle: r_efi::efi::Handle,
            protocol: *mut simple_text_output::Protocol,
        ) {
            unsafe {
                (*self.st.as_mut_ptr()).standard_error_handle = handle;
                (*self.st.as_mut_ptr()).std_err = protocol;
            }
        }

        pub fn stdout_init(&mut self, protocol: helpers::OwnedProtocol<PipeProtocol>) {
            self.set_stdout(
                protocol.handle().as_ptr(),
                protocol.as_ref() as *const PipeProtocol as *mut simple_text_output::Protocol,
            );
            self.stdout = Some(protocol);
        }

        pub fn stdout_inherit(&mut self) {
            let st: NonNull<r_efi::efi::SystemTable> = system_table().cast();
            unsafe { self.set_stdout((*st.as_ptr()).console_out_handle, (*st.as_ptr()).con_out) }
        }

        pub fn stderr_init(&mut self, protocol: helpers::OwnedProtocol<PipeProtocol>) {
            self.set_stderr(
                protocol.handle().as_ptr(),
                protocol.as_ref() as *const PipeProtocol as *mut simple_text_output::Protocol,
            );
            self.stderr = Some(protocol);
        }

        pub fn stderr_inherit(&mut self) {
            let st: NonNull<r_efi::efi::SystemTable> = system_table().cast();
            unsafe { self.set_stderr((*st.as_ptr()).standard_error_handle, (*st.as_ptr()).std_err) }
        }

        pub fn stderr(&self) -> io::Result<Vec<u8>> {
            match &self.stderr {
                Some(stderr) => stderr.as_ref().utf8(),
                None => Ok(Vec::new()),
            }
        }

        pub fn stdout(&self) -> io::Result<Vec<u8>> {
            match &self.stdout {
                Some(stdout) => stdout.as_ref().utf8(),
                None => Ok(Vec::new()),
            }
        }

        pub fn set_args(&mut self, args: Box<[u16]>) {
            let loaded_image: NonNull<loaded_image::Protocol> =
                helpers::open_protocol(self.handle, loaded_image::PROTOCOL_GUID).unwrap();

            let len = args.len();
            let args_size: u32 = (len * crate::mem::size_of::<u16>()).try_into().unwrap();
            let ptr = Box::into_raw(args).as_mut_ptr();

            unsafe {
                (*loaded_image.as_ptr()).load_options = ptr as *mut crate::ffi::c_void;
                (*loaded_image.as_ptr()).load_options_size = args_size;
            }

            self.args = Some((ptr, len));
        }

        fn update_st_crc32(&mut self) -> io::Result<()> {
            let bt: NonNull<r_efi::efi::BootServices> = boot_services().unwrap().cast();
            let st_size = unsafe { (*self.st.as_ptr()).hdr.header_size as usize };
            let mut crc32: u32 = 0;

            // Set crc to 0 before calculation
            unsafe {
                (*self.st.as_mut_ptr()).hdr.crc32 = 0;
            }

            let r = unsafe {
                ((*bt.as_ptr()).calculate_crc32)(
                    self.st.as_mut_ptr() as *mut crate::ffi::c_void,
                    st_size,
                    &mut crc32,
                )
            };

            if r.is_error() {
                Err(io::Error::from_raw_os_error(r.as_usize()))
            } else {
                unsafe {
                    (*self.st.as_mut_ptr()).hdr.crc32 = crc32;
                }
                Ok(())
            }
        }
    }

    impl Drop for Image {
        fn drop(&mut self) {
            if let Some(bt) = boot_services() {
                let bt: NonNull<r_efi::efi::BootServices> = bt.cast();
                unsafe {
                    ((*bt.as_ptr()).unload_image)(self.handle.as_ptr());
                }
            }

            if let Some((ptr, len)) = self.args {
                let _ = unsafe { Box::from_raw(crate::ptr::slice_from_raw_parts_mut(ptr, len)) };
            }
        }
    }

    #[repr(C)]
    pub struct PipeProtocol {
        reset: simple_text_output::ProtocolReset,
        output_string: simple_text_output::ProtocolOutputString,
        test_string: simple_text_output::ProtocolTestString,
        query_mode: simple_text_output::ProtocolQueryMode,
        set_mode: simple_text_output::ProtocolSetMode,
        set_attribute: simple_text_output::ProtocolSetAttribute,
        clear_screen: simple_text_output::ProtocolClearScreen,
        set_cursor_position: simple_text_output::ProtocolSetCursorPosition,
        enable_cursor: simple_text_output::ProtocolEnableCursor,
        mode: *mut simple_text_output::Mode,
        _buffer: Vec<u16>,
    }

    impl PipeProtocol {
        pub fn new() -> Self {
            let mode = Box::new(simple_text_output::Mode {
                max_mode: 0,
                mode: 0,
                attribute: 0,
                cursor_column: 0,
                cursor_row: 0,
                cursor_visible: r_efi::efi::Boolean::FALSE,
            });
            Self {
                reset: Self::reset,
                output_string: Self::output_string,
                test_string: Self::test_string,
                query_mode: Self::query_mode,
                set_mode: Self::set_mode,
                set_attribute: Self::set_attribute,
                clear_screen: Self::clear_screen,
                set_cursor_position: Self::set_cursor_position,
                enable_cursor: Self::enable_cursor,
                mode: Box::into_raw(mode),
                _buffer: Vec::new(),
            }
        }

        pub fn null() -> Self {
            let mode = Box::new(simple_text_output::Mode {
                max_mode: 0,
                mode: 0,
                attribute: 0,
                cursor_column: 0,
                cursor_row: 0,
                cursor_visible: r_efi::efi::Boolean::FALSE,
            });
            Self {
                reset: Self::reset_null,
                output_string: Self::output_string_null,
                test_string: Self::test_string,
                query_mode: Self::query_mode,
                set_mode: Self::set_mode,
                set_attribute: Self::set_attribute,
                clear_screen: Self::clear_screen,
                set_cursor_position: Self::set_cursor_position,
                enable_cursor: Self::enable_cursor,
                mode: Box::into_raw(mode),
                _buffer: Vec::new(),
            }
        }

        pub fn utf8(&self) -> io::Result<Vec<u8>> {
            OsString::from_wide(&self._buffer)
                .into_string()
                .map(Into::into)
                .map_err(|_| const_error!(io::ErrorKind::Other, "utf8 conversion failed"))
        }

        extern "efiapi" fn reset(
            proto: *mut simple_text_output::Protocol,
            _: r_efi::efi::Boolean,
        ) -> r_efi::efi::Status {
            let proto: *mut PipeProtocol = proto.cast();
            unsafe {
                (*proto)._buffer.clear();
            }
            r_efi::efi::Status::SUCCESS
        }

        extern "efiapi" fn reset_null(
            _: *mut simple_text_output::Protocol,
            _: r_efi::efi::Boolean,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::SUCCESS
        }

        extern "efiapi" fn output_string(
            proto: *mut simple_text_output::Protocol,
            buf: *mut r_efi::efi::Char16,
        ) -> r_efi::efi::Status {
            let proto: *mut PipeProtocol = proto.cast();
            let buf_len = unsafe {
                if let Some(x) = WStrUnits::new(buf) {
                    x.count()
                } else {
                    return r_efi::efi::Status::INVALID_PARAMETER;
                }
            };
            let buf_slice = unsafe { slice::from_raw_parts(buf, buf_len) };

            unsafe {
                (*proto)._buffer.extend_from_slice(buf_slice);
            };

            r_efi::efi::Status::SUCCESS
        }

        extern "efiapi" fn output_string_null(
            _: *mut simple_text_output::Protocol,
            _: *mut r_efi::efi::Char16,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::SUCCESS
        }

        extern "efiapi" fn test_string(
            _: *mut simple_text_output::Protocol,
            _: *mut r_efi::efi::Char16,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::SUCCESS
        }

        extern "efiapi" fn query_mode(
            _: *mut simple_text_output::Protocol,
            _: usize,
            _: *mut usize,
            _: *mut usize,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }

        extern "efiapi" fn set_mode(
            _: *mut simple_text_output::Protocol,
            _: usize,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }

        extern "efiapi" fn set_attribute(
            _: *mut simple_text_output::Protocol,
            _: usize,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }

        extern "efiapi" fn clear_screen(
            _: *mut simple_text_output::Protocol,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }

        extern "efiapi" fn set_cursor_position(
            _: *mut simple_text_output::Protocol,
            _: usize,
            _: usize,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }

        extern "efiapi" fn enable_cursor(
            _: *mut simple_text_output::Protocol,
            _: r_efi::efi::Boolean,
        ) -> r_efi::efi::Status {
            r_efi::efi::Status::UNSUPPORTED
        }
    }

    impl Drop for PipeProtocol {
        fn drop(&mut self) {
            unsafe {
                let _ = Box::from_raw(self.mode);
            }
        }
    }

    pub fn create_args(prog: &OsStr, args: &[OsString]) -> Box<[u16]> {
        const QUOTE: u16 = 0x0022;
        const SPACE: u16 = 0x0020;
        const CARET: u16 = 0x005e;
        const NULL: u16 = 0;

        // This is the lower bound on the final length under the assumption that
        // the arguments only contain ASCII characters.
        let mut res = Vec::with_capacity(args.iter().map(|arg| arg.len() + 3).sum());

        // Wrap program name in quotes to avoid any problems
        res.push(QUOTE);
        res.extend(prog.encode_wide());
        res.push(QUOTE);

        for arg in args {
            res.push(SPACE);

            // Wrap the argument in quotes to be treat as single arg
            res.push(QUOTE);
            for c in arg.encode_wide() {
                // CARET in quotes is used to escape CARET or QUOTE
                if c == QUOTE || c == CARET {
                    res.push(CARET);
                }
                res.push(c);
            }
            res.push(QUOTE);
        }

        res.into_boxed_slice()
    }
}
