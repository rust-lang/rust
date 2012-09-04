fn borrow<T>(x: &r/T) -> &r/T {x}

fn main() {
    let x = @3;
    loop {
        let y = borrow(x);
        assert *x == *y;
	break;
    }
}
