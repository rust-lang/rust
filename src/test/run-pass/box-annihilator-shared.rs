extern mod rustrt {
    #[legacy_exports];
    fn rust_annihilate_box(ptr: *uint);
}

fn main() {
    unsafe {
        let x = @3;
        let p: *uint = cast::transmute(x);
        rustrt::rust_annihilate_box(p);
    }
}
