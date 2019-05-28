// Reject mixing cyclic structure and Drop when using trait
// objects to hide the cross-references.
//
// (Compare against compile-fail/dropck_vec_cycle_checked.rs)

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
    use s;
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

trait Obj<'a> : HasId {
    fn set0(&self, b: &'a Box<dyn Obj<'a>>);
    fn set1(&self, b: &'a Box<dyn Obj<'a>>);
}

struct O<'a> {
    id: Id,
    obj0: CheckId<Cell<Option<&'a Box<dyn Obj<'a>>>>>,
    obj1: CheckId<Cell<Option<&'a Box<dyn Obj<'a>>>>>,
}

impl<'a> HasId for O<'a> {
    fn count(&self) -> usize { self.id.count() }
}

impl<'a> O<'a> {
    fn new() -> Box<O<'a>> {
        Box::new(O {
            id: Id::new(),
            obj0: CheckId(Cell::new(None)),
            obj1: CheckId(Cell::new(None)),
        })
    }
}

impl<'a> HasId for Cell<Option<&'a Box<dyn Obj<'a>>>> {
    fn count(&self) -> usize {
        match self.get() {
            None => 1,
            Some(c) => c.count(),
        }
    }
}

impl<'a> Obj<'a> for O<'a> {
    fn set0(&self, b: &'a Box<dyn Obj<'a>>) {
        self.obj0.v.set(Some(b))
    }
    fn set1(&self, b: &'a Box<dyn Obj<'a>>) {
        self.obj1.v.set(Some(b))
    }
}


fn f() {
    let (o1, o2, o3): (Box<dyn Obj>, Box<dyn Obj>, Box<dyn Obj>) = (O::new(), O::new(), O::new());
    o1.set0(&o2); //~ ERROR `o2` does not live long enough
    o1.set1(&o3); //~ ERROR `o3` does not live long enough
    o2.set0(&o2); //~ ERROR `o2` does not live long enough
    o2.set1(&o3); //~ ERROR `o3` does not live long enough
    o3.set0(&o1); //~ ERROR `o1` does not live long enough
    o3.set1(&o2); //~ ERROR `o2` does not live long enough
}

fn main() {
    f();
}
