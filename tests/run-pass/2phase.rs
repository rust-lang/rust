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
            if b { x = &mut y };
            22
        },
        x.len(),
    ));
}

/*
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
*/

fn with_interior_mutability() {
    use std::cell::Cell;

    trait Thing: Sized {
        fn do_the_thing(&mut self, _s: i32) {}
    }

    impl<T> Thing for Cell<T> {}

    let mut x = Cell::new(1);
    let l = &x;

    #[allow(unknown_lints, mutable_borrow_reservation_conflict)]
    x
        .do_the_thing({
            x.set(3);
            l.set(4);
            x.get() + l.get()
        })
    ;
}

fn main() {
    two_phase1();
    two_phase2();
    two_phase3(false);
    two_phase3(true);
    with_interior_mutability();
    //FIXME: enable these, or remove them, depending on how https://github.com/rust-lang/rust/issues/56254 gets resolved
    //two_phase_overlapping1();
    //two_phase_overlapping2();
}
