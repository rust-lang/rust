use sys::process::RawFd;

pub enum Stdio {
    MakePipe,
    Raw(RawFd),
    Inherit,
    None,
}
