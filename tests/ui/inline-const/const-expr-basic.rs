//@ run-pass

fn foo() -> i32 {
    const {
        let x = 5 + 10;
        x / 3
    }
}

fn main() {
    assert_eq!(5, foo());
}
