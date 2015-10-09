pub use imp::process as imp;

pub mod traits {
    pub use super::{Process as sys_Process, Command as sys_Command, ExitStatus as sys_ExitStatus};
}

pub mod prelude {
    pub use super::imp::Process;
    pub use super::traits::*;
    pub use super::Stdio;

    pub type RawFd = <Process as sys_Process>::RawFd;
    pub type Command = <Process as sys_Process>::Command;
    pub type ExitStatus = <Process as sys_Process>::ExitStatus;
    pub type PipeRead = <Process as sys_Process>::PipeRead;
    pub type PipeWrite = <Process as sys_Process>::PipeWrite;
}

use os_str::prelude::*;
use error::prelude::*;
use io;
use core::fmt;
use core::iter;

pub enum Stdio<P: Process> {
    MakePipe,
    Raw(P::RawFd),
    Inherit,
    None,
}

impl<P: Process> Clone for Stdio<P> where P::RawFd: Clone {
    fn clone(&self) -> Self {
        match *self {
            Stdio::MakePipe => Stdio::MakePipe,
            Stdio::Inherit => Stdio::Inherit,
            Stdio::None => Stdio::None,
            Stdio::Raw(ref fd) => Stdio::Raw(fd.clone()),
        }
    }
}

pub trait Command: Clone + fmt::Debug {
    fn new(program: &OsStr) -> Result<Self> where Self: Sized;

    fn arg(&mut self, arg: &OsStr);
    fn args<'a, I: iter::Iterator<Item = &'a OsStr>>(&mut self, args: I);
    fn env(&mut self, key: &OsStr, val: &OsStr);
    fn env_remove(&mut self, key: &OsStr);
    fn env_clear(&mut self);
    fn cwd(&mut self, dir: &OsStr);
}

pub trait ExitStatus: Sized + PartialEq + Eq + Clone + Copy + fmt::Debug + fmt::Display {
    fn success(&self) -> bool;
    fn code(&self) -> Option<i32>;
}

pub trait Process {
    type RawFd: Sized;
    type Command: Command;
    type ExitStatus: ExitStatus;
    type PipeRead: io::Read;
    type PipeWrite: io::Write;

    fn spawn(cfg: &Self::Command, stdin: Stdio<Self>, stdout: Stdio<Self>, stderr: Stdio<Self>) -> Result<Self> where Self: Sized;
    fn exit(code: i32) -> !;

    unsafe fn kill(&self) -> Result<()>;
    fn id(&self) -> Result<u32>;
    fn wait(&self) -> Result<Self::ExitStatus>;
    fn try_wait(&self) -> Option<Self::ExitStatus>;

    fn stdin(&mut self) -> &mut Option<Self::PipeWrite>;
    fn stdout(&mut self) -> &mut Option<Self::PipeRead>;
    fn stderr(&mut self) -> &mut Option<Self::PipeRead>;
}
