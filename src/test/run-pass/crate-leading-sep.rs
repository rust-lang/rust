// pretty-expanded FIXME #23616

fn main() {
    use ::std::mem;
    mem::drop(2_usize);
}
