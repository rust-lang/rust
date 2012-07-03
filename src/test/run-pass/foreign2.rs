

#[abi = "cdecl"]
extern mod rustrt {
    fn unsupervise();
}

#[abi = "cdecl"]
#[nolink]
extern mod bar { }

#[abi = "cdecl"]
#[nolink]
extern mod zed { }

#[abi = "cdecl"]
#[nolink]
extern mod libc {
    fn write(fd: int, buf: *u8,
             count: core::libc::size_t) -> core::libc::ssize_t;
}

#[abi = "cdecl"]
#[nolink]
extern mod baz { }

fn main(args: ~[str]) { }
