// FIXME: this miscompiles with optimizations, see <https://github.com/rust-lang/rust/issues/132898>.
//@compile-flags: -Zmir-opt-level=0

trait S: Sized {
    fn tpb(&mut self, _s: Self) {}
}

impl S for i32 {}

fn two_phase1() {
    let mut x = 3;
    x.tpb(x);
}

fn two_phase2() {
    let mut v = vec![];
    v.push(v.len());
}

fn two_phase3(b: bool) {
    let mut x = &mut vec![];
    let mut y = vec![];
    x.push((
        {
            if b {
                x = &mut y;
            }
            22
        },
        x.len(),
    ));
}

#[allow(unreachable_code)]
fn two_phase_raw() {
    let x: &mut Vec<i32> = &mut vec![];
    x.push({
        // Unfortunately this does not trigger the problem of creating a
        // raw ponter from a pointer that had a two-phase borrow derived from
        // it because of the implicit &mut reborrow.
        let raw = x as *mut _;
        unsafe {
            *raw = vec![1];
        }
        return;
    });
}

fn two_phase_overlapping1() {
    let mut x = vec![];
    let p = &x;
    x.push(p.len());
}

fn two_phase_overlapping2() {
    use std::ops::AddAssign;
    let mut x = 1;
    let l = &x;
    x.add_assign(x + *l);
}

fn with_interior_mutability() {
    use std::cell::Cell;

    trait Thing: Sized {
        fn do_the_thing(&mut self, _s: i32) {}
    }

    impl<T> Thing for Cell<T> {}

    let mut x = Cell::new(1);
    let l = &x;

    x.do_the_thing({
        x.set(3);
        l.set(4);
        x.get() + l.get()
    });
}

// This one really shouldn't be accepted, but since we treat 2phase as raw, we do accept it.
// Tree Borrows rejects it.
fn aliasing_violation() {
    struct Foo(u64);
    impl Foo {
        fn add(&mut self, n: u64) -> u64 {
            self.0 + n
        }
    }

    let mut f = Foo(0);
    let alias = &mut f.0 as *mut u64;
    let res = f.add(unsafe {
        *alias = 42;
        0
    });
    assert_eq!(res, 42);
}

fn main() {
    two_phase1();
    two_phase2();
    two_phase3(false);
    two_phase3(true);
    two_phase_raw();
    with_interior_mutability();
    two_phase_overlapping1();
    two_phase_overlapping2();
    aliasing_violation();
}
