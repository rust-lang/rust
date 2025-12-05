// Reject mixing cyclic structure and Drop when using fixed length
// arrays.
//
// (Compare against ui/span/dropck_vec_cycle_checked.rs)



use std::cell::Cell;
use id::Id;

mod s {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static S_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub fn next_count() -> usize {
        S_COUNT.fetch_add(1, Ordering::SeqCst) + 1
    }
}

mod id {
    use crate::s;
    #[derive(Debug)]
    pub struct Id {
        orig_count: usize,
        count: usize,
    }

    impl Id {
        pub fn new() -> Id {
            let c = s::next_count();
            println!("building Id {}", c);
            Id { orig_count: c, count: c }
        }
        pub fn count(&self) -> usize {
            println!("Id::count on {} returns {}", self.orig_count, self.count);
            self.count
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            println!("dropping Id {}", self.count);
            self.count = 0;
        }
    }
}

trait HasId {
    fn count(&self) -> usize;
}

#[derive(Debug)]
struct CheckId<T:HasId> {
    v: T
}

#[allow(non_snake_case)]
fn CheckId<T:HasId>(t: T) -> CheckId<T> { CheckId{ v: t } }

impl<T:HasId> Drop for CheckId<T> {
    fn drop(&mut self) {
        assert!(self.v.count() > 0);
    }
}

#[derive(Debug)]
struct B<'a> {
    id: Id,
    a: [CheckId<Cell<Option<&'a B<'a>>>>; 2]
}

impl<'a> HasId for Cell<Option<&'a B<'a>>> {
    fn count(&self) -> usize {
        match self.get() {
            None => 1,
            Some(b) => b.id.count(),
        }
    }
}

impl<'a> B<'a> {
    fn new() -> B<'a> {
        B { id: Id::new(), a: [CheckId(Cell::new(None)), CheckId(Cell::new(None))] }
    }
}

fn f() {
    let (b1, b2, b3);
    b1 = B::new();
    b2 = B::new();
    b3 = B::new();
    b1.a[0].v.set(Some(&b2));
    //~^ ERROR `b2` does not live long enough
    b1.a[1].v.set(Some(&b3));
    //~^ ERROR `b3` does not live long enough
    b2.a[0].v.set(Some(&b2));
    b2.a[1].v.set(Some(&b3));
    b3.a[0].v.set(Some(&b1));
    //~^ ERROR `b1` does not live long enough
    b3.a[1].v.set(Some(&b2));
}

fn main() {
    f();
}
