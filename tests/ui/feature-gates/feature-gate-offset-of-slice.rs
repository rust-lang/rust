use std::mem::offset_of;

struct S {
    a: u8,
    b: (u8, u8),
    c: [i32],
}

struct T {
    x: i32,
    y: S,
}

type Tup = (i32, [i32]);

fn main() {
    offset_of!(S, c); //~ ERROR size for values of type `[i32]` cannot be known at compilation time
}

fn other() {
    offset_of!(T, y); //~ ERROR size for values of type `[i32]` cannot be known at compilation time
}

fn tuple() {
    offset_of!(Tup, 1); //~ ERROR size for values of type `[i32]` cannot be known at compilation time
}
