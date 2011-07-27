

native "rust" mod rustrt {
    type vbuf;
    fn vec_buf[T](v: vec[T], offset: uint) -> vbuf;
}

native "rust" mod bar = "" { }

native "cdecl" mod zed = "" { }

native "cdecl" mod libc = "" {
    fn write(fd: int, buf: rustrt::vbuf, count: uint) -> int;
}

native "cdecl" mod baz = "" { }

fn main(args: vec[str]) { }