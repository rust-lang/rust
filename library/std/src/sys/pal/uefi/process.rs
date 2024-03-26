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

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    prog: OsString,
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
        Command { prog: program.to_os_string() }
    }

    pub fn arg(&mut self, _arg: &OsStr) {
        panic!("unsupported")
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

    pub fn stdout(&mut self, _stdout: Stdio) {
        panic!("unsupported")
    }

    pub fn stderr(&mut self, _stderr: Stdio) {
        panic!("unsupported")
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

    pub fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        let cmd = uefi_command_internal::Command::load_image(&self.prog)?;
        let stat = cmd.start_image()?;
        Ok((ExitStatus(stat), Vec::new(), Vec::new()))
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
    use super::super::helpers;
    use crate::ffi::OsStr;
    use crate::io::{self, const_io_error};
    use crate::mem::MaybeUninit;
    use crate::os::uefi::env::{boot_services, image_handle};
    use crate::ptr::NonNull;

    pub struct Command {
        handle: NonNull<crate::ffi::c_void>,
    }

    impl Command {
        const fn new(handle: NonNull<crate::ffi::c_void>) -> Self {
            Self { handle }
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
                Ok(Self::new(child_handle))
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
}
