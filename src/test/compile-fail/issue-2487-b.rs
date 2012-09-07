struct socket {
    sock: int,

    drop { }

    fn set_identity()  {
        do closure {
        setsockopt_bytes(self.sock) //~ ERROR copying a noncopyable value
      } 
    }
}

fn closure(f: fn@()) { f() }

fn setsockopt_bytes(+_sock: int) { }

fn main() {}
