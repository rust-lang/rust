// compile-pass

fn main() {}

const fn tup((a, b): (i32, i32)) -> i32 {
    a + b
}

const fn array([a, b]: [i32; 2]) -> i32 {
    a + b
}
