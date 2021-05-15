// See issue #12534.

fn main() {}

struct A(Box<u8>);

fn f(a @ A(u): A) -> Box<u8> {
    //~^ ERROR use of partially moved value
    drop(a);
    u
}
