

native "c-stack-cdecl" mod rustrt {
    fn unsupervise();
}

native "c-stack-cdecl" mod bar = "" { }

native "c-stack-cdecl" mod zed = "" { }

native "c-stack-cdecl" mod libc = "" {
    fn write(fd: int, buf: *u8, count: uint) -> int;
}

native "c-stack-cdecl" mod baz = "" { }

fn main(args: [str]) { }
