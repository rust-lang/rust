

native "rust" mod rustrt {
    type vbuf;
    fn vec_buf[T](vec[T] v, uint offset) -> vbuf;
}

native "rust" mod bar = "c" { }

native "cdecl" mod zed = "c" { }

native "cdecl" mod libc = "c" {
    fn write(int fd, rustrt::vbuf buf, uint count) -> int;
}

native "cdecl" mod baz = "c" { }

fn main(vec[str] args) { }