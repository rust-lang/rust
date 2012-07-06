fn borrow<T>(x: &T) -> &T {x}

fn main() {
    let x = @3;
    loop {
        let y = borrow(x);
        assert *x == *y;
	break;
    }
}
