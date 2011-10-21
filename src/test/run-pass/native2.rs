

native "c-stack-cdecl" mod rustrt {
    fn unsupervise();
}

native "c-stack-cdecl" mod bar = "" { }

native "cdecl" mod zed = "" { }

native "cdecl" mod libc = "" {
    fn write(fd: int, buf: *u8, count: uint) -> int;
}

native "cdecl" mod baz = "" { }

fn main(args: [str]) { }
