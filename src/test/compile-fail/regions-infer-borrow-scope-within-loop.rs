fn borrow<T>(x: &T) -> &T {x}

fn foo(cond: fn() -> bool, box: fn() -> @int) {
    let mut y: &int;
    loop {
        let x = box();

	// Here we complain because the resulting region
	// of this borrow is the fn body as a whole.
        y = borrow(x); //~ ERROR managed value would have to be rooted for lifetime 

        assert *x == *y;
        if cond() { break; }
    }
    assert *y != 0;
}

fn main() {}
