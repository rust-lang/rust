

native "rust" mod rustrt {
    type vbuf;
    fn vec_buf[T](vec[T] v, uint offset) -> vbuf;
}

native "rust" mod bar = "" { }

native "cdecl" mod zed = "" { }

native "cdecl" mod libc = "" {
    fn write(int fd, rustrt::vbuf buf, uint count) -> int;
}

native "cdecl" mod baz = "" { }

fn main(vec[str] args) { }