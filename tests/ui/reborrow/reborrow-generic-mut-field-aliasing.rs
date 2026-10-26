#![feature(reborrow)]

struct Wrap<'a, T>(&'a (), T);

impl<'a> std::marker::Reborrow for Wrap<'a, &'static mut i32> {}

fn get<'a, 'b>(x: Wrap<'a, &'b mut i32>) -> &'b mut i32 {
    x.1
}

fn main() {
    let mut x = 1;
    let thing = Wrap(&(), &mut x);
    let y = get(thing);
    let z = get(thing); //~ ERROR cannot borrow `thing` as mutable more than once at a time
    *y = 2;
    *z = 3;
}
