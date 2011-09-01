

native "rust" mod rustrt {
    fn unsupervise();
}

native "rust" mod bar = "" { }

native "cdecl" mod zed = "" { }

native "cdecl" mod libc = "" {
    fn write(fd: int, buf: *u8, count: uint) -> int;
}

native "cdecl" mod baz = "" { }

fn main(args: [istr]) { }
