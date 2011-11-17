

#[abi = "cdecl"]
native mod rustrt {
    fn unsupervise();
}

#[abi = "cdecl"]
#[link_name = ""]
native mod bar { }

#[abi = "cdecl"]
#[link_name = ""]
native mod zed { }

#[abi = "cdecl"]
#[link_name = ""]
native mod libc {
    fn write(fd: int, buf: *u8, count: uint) -> int;
}

#[abi = "cdecl"]
#[link_name = ""]
native mod baz { }

fn main(args: [str]) { }
