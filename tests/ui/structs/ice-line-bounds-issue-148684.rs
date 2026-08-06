struct A {
    b: Vec<u8>,
    c: usize,
}

fn main() {
    A(2, vec![])
    //~^ ERROR cannot find function, tuple struct or tuple variant `A` in this scope
}
