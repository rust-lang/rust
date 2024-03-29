use r_efi::protocols::simple_text_output;

use crate::ffi::OsStr;
use crate::ffi::OsString;
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::num::NonZero;
use crate::num::NonZeroI32;
use crate::path::Path;
use crate::sys::fs::File;
use crate::sys::pipe::AnonPipe;
use crate::sys::unsupported;
use crate::sys_common::process::{CommandEnv, CommandEnvs};

pub use crate::ffi::OsString as EnvKey;

use super::helpers;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    prog: OsString,
    args: OsString,
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

// FIXME: This should be a unit struct, so we can always construct it
// The value here should be never used, since we cannot spawn processes.
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            prog: program.to_os_string(),
            args: program.to_os_string(),
            stdout: None,
            stderr: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(" ");
        self.args.push(arg);
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
        panic!("unsupported")
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        panic!("unsupported")
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
    ) -> io::Result<Option<helpers::Protocol<uefi_command_internal::PipeProtocol>>> {
        match s {
            Stdio::MakePipe => helpers::Protocol::create(
                uefi_command_internal::PipeProtocol::new(),
                simple_text_output::PROTOCOL_GUID,
            )
            .map(Some),
            Stdio::Null => helpers::Protocol::create(
                uefi_command_internal::PipeProtocol::null(),
                simple_text_output::PROTOCOL_GUID,
            )
            .map(Some),
            Stdio::Inherit => Ok(None),
        }
    }

    pub fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        let mut cmd = uefi_command_internal::Command::load_image(&self.prog)?;

        let stdout: Option<helpers::Protocol<uefi_command_internal::PipeProtocol>> =
            match self.stdout.take() {
                Some(s) => Self::create_pipe(s),
                None => helpers::Protocol::create(
                    uefi_command_internal::PipeProtocol::new(),
                    simple_text_output::PROTOCOL_GUID,
                )
                .map(Some),
            }?;

        let stderr: Option<helpers::Protocol<uefi_command_internal::PipeProtocol>> =
            match self.stderr.take() {
                Some(s) => Self::create_pipe(s),
                None => helpers::Protocol::create(
                    uefi_command_internal::PipeProtocol::new(),
                    simple_text_output::PROTOCOL_GUID,
                )
                .map(Some),
            }?;

        match stdout {
            Some(stdout) => cmd.stdout_init(stdout),
            None => cmd.stdout_inherit(),
        };
        match stderr {
            Some(stderr) => cmd.stderr_init(stderr),
            None => cmd.stderr_inherit(),
        };

        if !self.args.is_empty() {
            cmd.set_args(&self.args);
        }

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

impl fmt::Debug for Command {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
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
    _p: PhantomData<&'a ()>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}

mod uefi_command_internal {
    use r_efi::protocols::{loaded_image, simple_text_output};

    use super::super::helpers;
    use crate::ffi::{OsStr, OsString};
    use crate::io::{self, const_io_error};
    use crate::mem::MaybeUninit;
    use crate::os::uefi::env::{boot_services, image_handle, system_table};
    use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
    use crate::ptr::NonNull;
    use crate::slice;
    use crate::sys_common::wstr::WStrUnits;

    pub struct Command {
        handle: NonNull<crate::ffi::c_void>,
        stdout: Option<helpers::Protocol<PipeProtocol>>,
        stderr: Option<helpers::Protocol<PipeProtocol>>,
        st: Box<r_efi::efi::SystemTable>,
        args: Option<Vec<u16>>,
    }

    impl Command {
        const fn new(
            handle: NonNull<crate::ffi::c_void>,
            st: Box<r_efi::efi::SystemTable>,
        ) -> Self {
            Self { handle, stdout: None, stderr: None, st, args: None }
        }

        pub fn load_image(p: &OsStr) -> io::Result<Self> {
            let mut path = helpers::DevicePath::from_text(p)?;
            let boot_services: NonNull<r_efi::efi::BootServices> = boot_services()
                .ok_or_else(|| const_io_error!(io::ErrorKind::NotFound, "Boot Services not found"))?
                .cast();
            let mut child_handle: MaybeUninit<r_efi::efi::Handle> = MaybeUninit::uninit();
            let image_handle = image_handle();

            let r = unsafe {
                ((*boot_services.as_ptr()).load_image)(
                    r_efi::efi::Boolean::FALSE,
                    image_handle.as_ptr(),
                    path.as_mut(),
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
                let mut st: Box<r_efi::efi::SystemTable> =
                    Box::new(unsafe { crate::ptr::read((*loaded_image.as_ptr()).system_table) });

                unsafe {
                    (*loaded_image.as_ptr()).system_table = st.as_mut();
                }

                Ok(Self::new(child_handle, st))
            }
        }

        pub fn start_image(&self) -> io::Result<r_efi::efi::Status> {
            let boot_services: NonNull<r_efi::efi::BootServices> = boot_services()
                .ok_or_else(|| const_io_error!(io::ErrorKind::NotFound, "Boot Services not found"))?
                .cast();
            let mut exit_data_size: MaybeUninit<usize> = MaybeUninit::uninit();
            let mut exit_data: MaybeUninit<*mut u16> = MaybeUninit::uninit();

            let r = unsafe {
                ((*boot_services.as_ptr()).start_image)(
                    self.handle.as_ptr(),
                    exit_data_size.as_mut_ptr(),
                    exit_data.as_mut_ptr(),
                )
            };

            // Drop exitdata
            unsafe {
                exit_data_size.assume_init_drop();
                exit_data.assume_init_drop();
            }

            Ok(r)
        }

        pub fn stdout_init(&mut self, mut protocol: helpers::Protocol<PipeProtocol>) {
            self.st.console_out_handle = protocol.handle().as_ptr();
            self.st.con_out =
                protocol.as_mut() as *mut PipeProtocol as *mut simple_text_output::Protocol;

            self.stdout = Some(protocol);
        }

        pub fn stdout_inherit(&mut self) {
            let st: NonNull<r_efi::efi::SystemTable> = system_table().cast();

            self.st.console_out_handle = unsafe { (*st.as_ptr()).console_out_handle };
            self.st.con_out = unsafe { (*st.as_ptr()).con_out };
        }

        pub fn stderr_init(&mut self, mut protocol: helpers::Protocol<PipeProtocol>) {
            self.st.standard_error_handle = protocol.handle().as_ptr();
            self.st.std_err =
                protocol.as_mut() as *mut PipeProtocol as *mut simple_text_output::Protocol;

            self.stderr = Some(protocol);
        }

        pub fn stderr_inherit(&mut self) {
            let st: NonNull<r_efi::efi::SystemTable> = system_table().cast();

            self.st.standard_error_handle = unsafe { (*st.as_ptr()).standard_error_handle };
            self.st.std_err = unsafe { (*st.as_ptr()).std_err };
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

        pub fn set_args(&mut self, args: &OsStr) {
            let loaded_image: NonNull<loaded_image::Protocol> =
                helpers::open_protocol(self.handle, loaded_image::PROTOCOL_GUID).unwrap();

            let mut args = args.encode_wide().collect::<Vec<u16>>();
            let args_size = (crate::mem::size_of::<u16>() * args.len()) as u32;

            unsafe {
                (*loaded_image.as_ptr()).load_options =
                    args.as_mut_ptr() as *mut crate::ffi::c_void;
                (*loaded_image.as_ptr()).load_options_size = args_size;
            }

            self.args = Some(args);
        }
    }

    impl Drop for Command {
        fn drop(&mut self) {
            if let Some(bt) = boot_services() {
                let bt: NonNull<r_efi::efi::BootServices> = bt.cast();
                unsafe {
                    ((*bt.as_ptr()).unload_image)(self.handle.as_ptr());
                }
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
        _mode: Box<simple_text_output::Mode>,
        _buffer: Vec<u16>,
    }

    impl PipeProtocol {
        pub fn new() -> Self {
            let mut mode = Box::new(simple_text_output::Mode {
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
                mode: mode.as_mut(),
                _mode: mode,
                _buffer: Vec::new(),
            }
        }

        pub fn null() -> Self {
            let mut mode = Box::new(simple_text_output::Mode {
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
                mode: mode.as_mut(),
                _mode: mode,
                _buffer: Vec::new(),
            }
        }

        pub fn utf8(&self) -> io::Result<Vec<u8>> {
            OsString::from_wide(&self._buffer)
                .into_string()
                .map(Into::into)
                .map_err(|_| const_io_error!(io::ErrorKind::Other, "utf8 conversion failed"))
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
}
