//@ run-pass
#![allow(path_statements)]
#![allow(dead_code)]
// A battery of tests to ensure destructors of unboxed closure environments
// run at the right times.

static mut DROP_COUNT: usize = 0;

fn drop_count() -> usize {
    unsafe {
        DROP_COUNT
    }
}

struct Droppable {
    x: isize,
}

impl Droppable {
    fn new() -> Droppable {
        Droppable {
            x: 1
        }
    }
}

impl Drop for Droppable {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1
        }
    }
}

fn a<F:Fn(isize, isize) -> isize>(f: F) -> isize {
    f(1, 2)
}

fn b<F:FnMut(isize, isize) -> isize>(mut f: F) -> isize {
    f(3, 4)
}

fn c<F:FnOnce(isize, isize) -> isize>(f: F) -> isize {
    f(5, 6)
}

fn test_fn() {
    {
        a(move |a: isize, b| { a + b });
    }
    assert_eq!(drop_count(), 0);

    {
        let z = &Droppable::new();
        a(move |a: isize, b| { z; a + b });
        assert_eq!(drop_count(), 0);
    }
    assert_eq!(drop_count(), 1);

    {
        let z = &Droppable::new();
        let zz = &Droppable::new();
        a(move |a: isize, b| { z; zz; a + b });
        assert_eq!(drop_count(), 1);
    }
    assert_eq!(drop_count(), 3);
}

fn test_fn_mut() {
    {
        b(move |a: isize, b| { a + b });
    }
    assert_eq!(drop_count(), 3);

    {
        let z = &Droppable::new();
        b(move |a: isize, b| { z; a + b });
        assert_eq!(drop_count(), 3);
    }
    assert_eq!(drop_count(), 4);

    {
        let z = &Droppable::new();
        let zz = &Droppable::new();
        b(move |a: isize, b| { z; zz; a + b });
        assert_eq!(drop_count(), 4);
    }
    assert_eq!(drop_count(), 6);
}

fn test_fn_once() {
    {
        c(move |a: isize, b| { a + b });
    }
    assert_eq!(drop_count(), 6);

    {
        let z = Droppable::new();
        c(move |a: isize, b| { z; a + b });
        assert_eq!(drop_count(), 7);
    }
    assert_eq!(drop_count(), 7);

    {
        let z = Droppable::new();
        let zz = Droppable::new();
        c(move |a: isize, b| { z; zz; a + b });
        assert_eq!(drop_count(), 9);
    }
    assert_eq!(drop_count(), 9);
}

fn main() {
    test_fn();
    test_fn_mut();
    test_fn_once();
}
