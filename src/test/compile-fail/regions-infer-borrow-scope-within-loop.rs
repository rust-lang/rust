fn borrow<T>(x: &T) -> &T {x}

fn main() {
    let x = @3;
    let y: &int; //~ ERROR reference is not valid outside of its lifetime
    while true {
        y = borrow(x);
        assert *x == *y;
    }
    assert *x == *y;
}
