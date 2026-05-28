fn id<T>(x: T) -> T { x }

fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block:
    let mut x;

    if cond {
        x = &id(3); //~ ERROR temporary value dropped while borrowed
        assert_eq!(*x, 3);
    }
}

fn main() {}
