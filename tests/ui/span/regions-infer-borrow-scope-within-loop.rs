fn borrow<T>(x: &T) -> &T {x}

fn foo<C, M>(mut cond: C, mut make_box: M) where
    C: FnMut() -> bool,
    M: FnMut() -> Box<isize>,
{
    let mut y: &isize;
    loop {
        let x = make_box();

        // Here we complain because the resulting region
        // of this borrow is the fn body as a whole.
        y = borrow(&*x);
        //~^ ERROR `*x` does not live long enough

        assert_eq!(*x, *y);
        if cond() { break; }
    }
    assert!(*y != 0);
}

fn main() {}
