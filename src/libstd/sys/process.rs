pub use sys::imp::process::{
    Process, Command, ExitStatus,
    RawFd, PipeRead, PipeWrite,
    spawn, exit,
};

pub enum Stdio {
    MakePipe,
    Raw(RawFd),
    Inherit,
    None,
}

impl Clone for Stdio {
    fn clone(&self) -> Self {
        match *self {
            Stdio::MakePipe => Stdio::MakePipe,
            Stdio::Inherit => Stdio::Inherit,
            Stdio::None => Stdio::None,
            Stdio::Raw(ref fd) => Stdio::Raw(fd.clone()),
        }
    }
}
