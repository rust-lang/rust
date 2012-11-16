fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block:
    let mut x;

    if cond {
        x = &3; //~ ERROR illegal borrow: borrowed value does not live long enough
        assert (*x == 3);
    }
}

fn main() {}