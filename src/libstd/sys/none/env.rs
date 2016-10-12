use io;

pub fn generic_error() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "environment variables not supported on this platform")
}

pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "none";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}
