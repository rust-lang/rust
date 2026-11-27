//@ run-pass

fn borrow<T>(x: &T) -> &T {x}

pub fn main() {
    let x: Box<_> = Box::new(3);
    loop {
        let y = borrow(&*x);
        assert_eq!(*x, *y);
        break;
    }
}
