fn main() {
    trait T { }
    #[derive(Debug)]
    struct S {
        x: * mut dyn T
    }
    dbg!(S { x: unsafe { std::mem::transmute((0usize, 0usize)) } }); //~ ERROR: encountered dangling or unaligned vtable pointer
}
