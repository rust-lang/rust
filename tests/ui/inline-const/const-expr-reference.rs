//@ run-pass

const fn bar() -> i32 {
    const {
        2 + 3
    }
}

fn main() {
    let x: &'static i32 = &const{bar()};
    assert_eq!(&5, x);
}
