// run-pass

trait MyInt {
    const ONE: Self;
}

impl MyInt for i32 {
    const ONE: i32 = 1;
}

fn main() {
    assert_eq!(1, <i32>::ONE);
}
