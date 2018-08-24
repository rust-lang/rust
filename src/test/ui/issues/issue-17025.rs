// ignore-test the unsized enum no longer compiles

enum A {
    B(char),
    C([Box<A>]),
}

fn c(c:char) {
    A::B(c);
    //~^ ERROR cannot move a value of type A: the size of A cannot be statically determined
}

pub fn main() {}
