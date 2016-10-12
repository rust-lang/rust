use io;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

fn generic_error() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "standard I/O not supported on this platform")
}

impl Stdin {
    pub fn new() -> io::Result<Stdin> { Err(generic_error()) }
    pub fn read(&self, _data: &mut [u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> { Err(generic_error()) }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> { Err(generic_error()) }
    pub fn write(&self, _data: &[u8]) -> io::Result<usize> { Err(generic_error()) }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> { Err(generic_error()) }
    pub fn write(&self, _data: &[u8]) -> io::Result<usize> { Err(generic_error()) }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, _data: &[u8]) -> io::Result<usize> { Err(generic_error()) }
    fn flush(&mut self) -> io::Result<()> { Err(generic_error()) }
}

pub const EBADF_ERR: i32 = 0;
