extern mod rustrt {
    fn rust_annihilate_box(ptr: *uint);
}

fn main() {
    unsafe {
        let x = ~[~"a", ~"b", ~"c"];
        let p: *uint = cast::transmute(x);
        rustrt::rust_annihilate_box(p);
    }
}
