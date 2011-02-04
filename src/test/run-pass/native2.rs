native "rust" mod rustrt {
    type vbuf;
    fn vec_buf[T](vec[T] v, uint offset) -> vbuf;
}

fn main(vec[str] args) {
}
