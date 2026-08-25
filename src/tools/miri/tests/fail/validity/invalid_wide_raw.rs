#![allow(invalid_value)]

fn main() {
    trait T {}
    #[derive(Debug)]
    struct S {
        #[allow(dead_code)]
        x: *mut dyn T,
    }
    dbg!(S { x: unsafe { std::mem::transmute((0usize, 0usize)) } }); //~ ERROR: encountered null pointer, but expected a vtable pointer
}
