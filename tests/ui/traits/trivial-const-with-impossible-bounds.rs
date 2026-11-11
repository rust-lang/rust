#![crate_type = "lib"]

struct Dummy;
impl Dummy where for<'a> &'a mut i32: Copy {
    const C: usize = 1; //~ ERROR entering unreachable code
}

fn foo() where for<'a> &'a mut i32: Copy {
    if let Dummy::C = 1 {}
}
