//@ build-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


struct socket {
    sock: isize,

}

impl Drop for socket {
    fn drop(&mut self) {}
}

impl socket {
    pub fn set_identity(&self)  {
        closure(|| setsockopt_bytes(self.sock.clone()))
    }
}

fn socket() -> socket {
    socket {
        sock: 1
    }
}

fn closure<F>(f: F) where F: FnOnce() { f() }

fn setsockopt_bytes(_sock: isize) { }

pub fn main() {}
