struct socket {
    sock: int,

    drop { }
}

impl socket {

    fn set_identity()  {
        do closure {
            setsockopt_bytes(copy self.sock)
        }
    }
}

fn socket() -> socket {
    socket {
        sock: 1
    }
}

fn closure(f: fn()) { f() }

fn setsockopt_bytes(_sock: int) { }

fn main() {}
