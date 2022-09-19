use crate::ffi::OsStr;
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::num::NonZeroI32;
use crate::path::Path;
use crate::ptr::NonNull;
use crate::sys::uefi::common;
use crate::sys::uefi::{common::status_to_io_error, fs::File, pipe::AnonPipe, unsupported};
use crate::sys_common::process::{CommandEnv, CommandEnvs};

pub use crate::ffi::OsString as EnvKey;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    env: CommandEnv,
    program: crate::ffi::OsString,
    args: crate::ffi::OsString,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
    stdin: Option<Stdio>,
}
// passed back to std::process with the pipes connected to the child, if any were requested
#[derive(Default)]
pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

#[derive(Clone, Copy)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        let program = super::path::absolute(&crate::path::PathBuf::from(program)).unwrap();
        Command {
            env: Default::default(),
            program: program.clone().into_os_string(),
            args: program.into_os_string(),
            stdout: None,
            stderr: None,
            stdin: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(" ");
        self.args.push(arg);
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }

    pub fn cwd(&mut self, _dir: &OsStr) {}

    pub fn stdin(&mut self, stdin: Stdio) {
        self.stdin = Some(stdin)
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout)
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr)
    }

    pub fn get_program(&self) -> &OsStr {
        self.program.as_os_str()
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        CommandArgs { _p: PhantomData }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        None
    }

    fn setup_pipe<F>(output: Stdio, func: F) -> io::Result<(r_efi::efi::Handle, Option<AnonPipe>)>
    where
        F: Fn(NonNull<uefi_command_protocol::Protocol>) -> r_efi::efi::Handle,
    {
        match output {
            Stdio::Inherit => {
                if let Some(command_protocol) =
                    common::get_current_handle_protocol::<uefi_command_protocol::Protocol>(
                        uefi_command_protocol::PROTOCOL_GUID,
                    )
                {
                    Ok((func(command_protocol), None))
                } else {
                    Ok((crate::ptr::null_mut(), None))
                }
            }
            Stdio::Null => {
                let pipe = AnonPipe::null();
                Ok((pipe.handle().as_ptr(), Some(pipe)))
            }
            Stdio::MakePipe => {
                let pipe = AnonPipe::make_pipe();
                Ok((pipe.handle().as_ptr(), Some(pipe)))
            }
        }
    }

    pub fn spawn(
        &mut self,
        default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        let mut cmd = uefi_command::Command::load_image(self.program.as_os_str())?;
        let mut command_protocol = uefi_command_protocol::Protocol::default();
        cmd.set_args(self.args.as_os_str())?;

        let mut stdio_pipe = StdioPipes::default();

        (command_protocol.stdout, stdio_pipe.stdout) =
            Self::setup_pipe(self.stdout.unwrap_or(default), |command_protocol| unsafe {
                (*command_protocol.as_ptr()).stdout
            })?;
        (command_protocol.stderr, stdio_pipe.stderr) =
            Self::setup_pipe(self.stderr.unwrap_or(default), |command_protocol| unsafe {
                (*command_protocol.as_ptr()).stderr
            })?;
        (command_protocol.stdin, stdio_pipe.stdin) =
            Self::setup_pipe(self.stdin.unwrap_or(default), |command_protocol| unsafe {
                (*command_protocol.as_ptr()).stdin
            })?;

        // Set env varibles
        for (key, val) in self.env.iter() {
            match val {
                None => crate::env::remove_var(key),
                Some(x) => crate::env::set_var(key, x),
            }
        }

        cmd.command_protocol = Some(common::ProtocolWrapper::install_protocol_in(
            command_protocol,
            cmd.handle.as_ptr(),
        )?);

        // Initially thought to implement start at wait. However, it seems like everything expectes
        // stdio output to be ready for reading before calling wait, which is not possible at least
        // in current implementation.
        // Moving this to wait breaks output
        let r = cmd.start_image()?;

        // Cleanup env
        for (k, _) in self.env.iter() {
            let _ = super::os::unsetenv(k);
        }

        let proc = Process { status: r };

        Ok((proc, stdio_pipe))
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        pipe.diverge()
    }
}

impl From<File> for Stdio {
    fn from(_file: File) -> Stdio {
        panic!("unsupported")
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.args.fmt(f)
    }
}

#[derive(Copy, PartialEq, Eq, Clone)]
pub struct ExitStatus(r_efi::efi::Status);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        if self.0.is_error() { Err(ExitStatusError(*self)) } else { Ok(()) }
    }

    pub fn code(&self) -> Option<i32> {
        Some(self.0.as_usize() as i32)
    }
}

impl fmt::Debug for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&status_to_io_error(self.0), f)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&status_to_io_error(self.0), f)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(ExitStatus);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        self.0
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZeroI32> {
        NonZeroI32::new(self.0.0.as_usize() as i32)
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

pub struct Process {
    status: r_efi::efi::Status,
}

impl Process {
    // Process does not have an id in UEFI
    pub fn id(&self) -> u32 {
        unimplemented!()
    }

    pub fn kill(&mut self) -> io::Result<()> {
        unsupported()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        Ok(ExitStatus(self.status))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        Ok(Some(ExitStatus(self.status)))
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
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}

mod uefi_command {
    use super::super::common::{self, status_to_io_error};
    use crate::ffi::OsStr;
    use crate::io;
    use crate::mem::{ManuallyDrop, MaybeUninit};
    use crate::os::uefi;
    use crate::ptr::{addr_of_mut, NonNull};
    use r_efi::protocols::loaded_image;

    pub(crate) struct Command {
        pub handle: NonNull<crate::ffi::c_void>,
        pub command_protocol:
            Option<common::ProtocolWrapper<super::uefi_command_protocol::Protocol>>,
    }

    impl Command {
        pub(crate) fn load_image(p: &OsStr) -> io::Result<Self> {
            let boot_services = common::get_boot_services().ok_or(common::BOOT_SERVICES_ERROR)?;
            let system_handle = uefi::env::image_handle();
            let mut path = super::super::path::device_path_from_os_str(p)?;
            let mut child_handle: MaybeUninit<r_efi::efi::Handle> = MaybeUninit::uninit();
            let r = unsafe {
                ((*boot_services.as_ptr()).load_image)(
                    r_efi::efi::Boolean::FALSE,
                    system_handle.as_ptr(),
                    path.as_mut(),
                    crate::ptr::null_mut(),
                    0,
                    child_handle.as_mut_ptr(),
                )
            };
            if r.is_error() {
                Err(status_to_io_error(r))
            } else {
                let child_handle = unsafe { child_handle.assume_init() };
                match NonNull::new(child_handle) {
                    None => {
                        Err(io::const_io_error!(io::ErrorKind::InvalidData, "null handle received"))
                    }
                    Some(x) => Ok(Self { handle: x, command_protocol: None }),
                }
            }
        }

        pub(crate) fn start_image(&self) -> io::Result<r_efi::efi::Status> {
            let boot_services = common::get_boot_services().ok_or(common::BOOT_SERVICES_ERROR)?;
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

        pub(crate) fn set_args(&self, args: &OsStr) -> io::Result<()> {
            let protocol: NonNull<loaded_image::Protocol> =
                common::open_protocol(self.handle, loaded_image::PROTOCOL_GUID)?;
            let mut args = ManuallyDrop::new(common::to_ffi_string(args));
            let args_size = (crate::mem::size_of::<u16>() * args.len()) as u32;
            unsafe {
                addr_of_mut!((*protocol.as_ptr()).load_options_size).write(args_size);
                addr_of_mut!((*protocol.as_ptr()).load_options).replace(args.as_mut_ptr().cast());
            };
            Ok(())
        }
    }

    impl Drop for Command {
        // Unload Image
        fn drop(&mut self) {
            if let Some(boot_services) = common::get_boot_services() {
                self.command_protocol = None;
                let _ = unsafe { ((*boot_services.as_ptr()).unload_image)(self.handle.as_ptr()) };
            }
        }
    }
}

pub(crate) mod uefi_command_protocol {
    use super::super::common;
    use r_efi::efi::{Guid, Handle};

    pub(crate) const PROTOCOL_GUID: Guid = Guid::from_fields(
        0xc3cc5ede,
        0xb029,
        0x4daa,
        0xa5,
        0x5f,
        &[0x93, 0xf8, 0x82, 0x5b, 0x29, 0xe7],
    );

    #[repr(C)]
    pub(crate) struct Protocol {
        pub stdout: Handle,
        pub stderr: Handle,
        pub stdin: Handle,
    }

    impl Default for Protocol {
        #[inline]
        fn default() -> Self {
            Self::new(crate::ptr::null_mut(), crate::ptr::null_mut(), crate::ptr::null_mut())
        }
    }

    impl Protocol {
        #[inline]
        pub(crate) fn new(stdout: Handle, stderr: Handle, stdin: Handle) -> Self {
            Self { stdout, stderr, stdin }
        }
    }

    impl common::Protocol for Protocol {
        const PROTOCOL_GUID: Guid = PROTOCOL_GUID;
    }
}
