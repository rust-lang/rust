struct A {
    b: Vec<u8>,
    c: usize,
}

fn main() {
    A(2, vec![])
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `A`
}
