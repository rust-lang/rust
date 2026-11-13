pub mod io {
    pub trait Read {
        fn read(&mut self);
    }
}

pub mod bufreader {
    //@ has crate_relative_assoc/bufreader/index.html '//a/@href' 'struct.TcpStream.html#method.read'
    //! [`crate::TcpStream::read`]
    use crate::io::Read;
}

pub struct TcpStream;

impl crate::io::Read for TcpStream {
    fn read(&mut self) {}
}
