fn borrow<T>(x: &r/T) -> &r/T {x}

fn foo(cond: fn() -> bool, box: fn() -> @int) {
    let mut y: &int;
    loop {
        let x = box();

	// Here we complain because the resulting region
	// of this borrow is the fn body as a whole.
        y = borrow(x); //~ ERROR illegal borrow: managed value would have to be rooted

        assert *x == *y;
        if cond() { break; }
    }
    assert *y != 0;
}

fn main() {}
