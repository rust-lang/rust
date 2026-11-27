// emit-mir
//@ check-pass

const fn foo() -> i32 {
    5 + 6
}

fn main() {
    foo();
}
