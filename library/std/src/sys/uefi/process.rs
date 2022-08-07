use crate::ffi::OsStr;
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::num::NonZeroI32;
use crate::path::Path;
use crate::sys::fs::File;
use crate::sys::pipe::AnonPipe;
use crate::sys::unsupported;
use crate::sys_common::process::{CommandEnv, CommandEnvs};

pub use crate::ffi::OsString as EnvKey;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    env: CommandEnv,
    program: crate::ffi::OsString,
    args: crate::ffi::OsString,
    stdout_key: Option<crate::ffi::OsString>,
    stderr_key: Option<crate::ffi::OsString>,
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
            stdout_key: None,
            stderr_key: None,
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
        match stdin {
            Stdio::Inherit => unimplemented!(),
            Stdio::Null => {
                let mut key = self.program.clone();
                key.push("_stdin");
                self.env.set(&key, OsStr::new("null"));
            }
            Stdio::MakePipe => unimplemented!(),
        }
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        match stdout {
            Stdio::Inherit => {
                if let Ok(current_exe) = crate::env::current_exe() {
                    let mut key = current_exe.into_os_string();
                    key.push("_stdout");
                    if let Some(val) = crate::env::var_os(&key) {
                        self.stdout_key = Some(val);
                    }
                }
            }
            Stdio::Null => {
                let mut key = self.program.clone();
                key.push("_stdout");
                self.env.set(&key, OsStr::new("null"));
            }
            Stdio::MakePipe => {
                let mut key = self.program.clone();
                key.push("_stdout");
                let mut val = self.program.clone();
                val.push("_stdout_pipe");
                self.env.set(&key, &val);
                self.stdout_key = Some(val);
            }
        }
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        match stderr {
            Stdio::Inherit => {
                if let Ok(current_exe) = crate::env::current_exe() {
                    let mut key = current_exe.into_os_string();
                    key.push("_stderr");
                    if let Some(val) = crate::env::var_os(&key) {
                        self.stderr_key = Some(val);
                    }
                }
            }
            Stdio::Null => {
                let mut key = self.program.clone();
                key.push("_stderr");
                self.env.set(&key, OsStr::new("null"));
            }
            Stdio::MakePipe => {
                let mut key = self.program.clone();
                key.push("_stderr");
                let mut val = self.program.clone();
                val.push("_stderr_pipe");
                self.env.set(&key, &val);
                self.stderr_key = Some(val);
            }
        }
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

    pub fn spawn(
        &mut self,
        default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        let cmd = uefi_command::Command::load_image(self.program.as_os_str())?;
        cmd.set_args(self.args.as_os_str())?;

        let mut stdio_pipe = StdioPipes::default();

        // Set defaults
        if self.stdout_key.is_none() {
            self.stdout(default);
        }
        if self.stderr_key.is_none() {
            self.stderr(default);
        }

        if let Some(x) = &self.stdout_key {
            stdio_pipe.stdout = Some(AnonPipe::new(x));
        }
        if let Some(x) = &self.stderr_key {
            stdio_pipe.stderr = Some(AnonPipe::new(x));
        }

        // Set env varibles
        for (key, val) in self.env.iter() {
            match val {
                None => crate::env::remove_var(key),
                Some(x) => crate::env::set_var(key, x),
            }
        }

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
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
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
        fmt::Debug::fmt(&super::common::status_to_io_error(self.0), f)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&super::common::status_to_io_error(self.0), f)
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
        unsupported()
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
    use crate::ffi::OsStr;
    use crate::io;
    use crate::mem::{ManuallyDrop, MaybeUninit};
    use crate::os::uefi;
    use crate::os::uefi::ffi::OsStrExt;
    use crate::ptr::NonNull;
    use r_efi::protocols::loaded_image;

    pub struct Command {
        inner: NonNull<crate::ffi::c_void>,
    }

    impl Command {
        pub fn load_image(p: &OsStr) -> io::Result<Self> {
            let boot_services = uefi::env::get_boot_services().ok_or(io::Error::new(
                io::ErrorKind::Uncategorized,
                "Failed to acquire boot_services",
            ))?;
            let system_handle = uefi::env::get_system_handle().ok_or(io::Error::new(
                io::ErrorKind::Uncategorized,
                "Failed to acquire System Handle",
            ))?;
            let path = uefi::path::DevicePath::try_from(p)?;
            let mut child_handle: MaybeUninit<r_efi::efi::Handle> = MaybeUninit::uninit();
            let r = unsafe {
                ((*boot_services.as_ptr()).load_image)(
                    r_efi::efi::Boolean::FALSE,
                    system_handle.as_ptr(),
                    path.as_ptr(),
                    crate::ptr::null_mut(),
                    0,
                    child_handle.as_mut_ptr(),
                )
            };
            if r.is_error() {
                Err(super::super::common::status_to_io_error(r))
            } else {
                let child_handle = unsafe { child_handle.assume_init() };
                match NonNull::new(child_handle) {
                    None => Err(io::Error::new(io::ErrorKind::InvalidData, "Null Handle Received")),
                    Some(x) => Ok(Self { inner: x }),
                }
            }
        }

        pub fn start_image(&self) -> io::Result<r_efi::efi::Status> {
            let boot_services = uefi::env::get_boot_services().ok_or(io::Error::new(
                io::ErrorKind::Uncategorized,
                "Failed to acquire boot_services",
            ))?;
            let mut exit_data_size: MaybeUninit<usize> = MaybeUninit::uninit();
            let mut exit_data: MaybeUninit<*mut u16> = MaybeUninit::uninit();
            let r = unsafe {
                ((*boot_services.as_ptr()).start_image)(
                    self.inner.as_ptr(),
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

        pub fn set_args(&self, args: &OsStr) -> io::Result<()> {
            let mut guid = loaded_image::PROTOCOL_GUID;
            let protocol: NonNull<loaded_image::Protocol> =
                uefi::env::get_handle_protocol(self.inner, &mut guid).ok_or(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    "Failed to acquire loaded image protocol for child handle",
                ))?;
            let mut args = ManuallyDrop::new(args.to_ffi_string());
            let args_size = (crate::mem::size_of::<u16>() * args.len()) as u32;
            unsafe {
                (*protocol.as_ptr()).load_options_size = args_size;
                let _ = crate::mem::replace(
                    &mut (*protocol.as_ptr()).load_options,
                    args.as_mut_ptr() as *mut crate::ffi::c_void,
                );
            };
            Ok(())
        }
    }

    impl Drop for Command {
        // Unload Image
        fn drop(&mut self) {
            if let Some(boot_services) = uefi::env::get_boot_services() {
                let _ = unsafe { ((*boot_services.as_ptr()).unload_image)(self.inner.as_ptr()) };
            }
        }
    }
}
