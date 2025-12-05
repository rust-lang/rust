// Checking that `Vec<T>` cannot hide lifetimes within `T` when `T`
// implements `Drop` and might access methods of values that have
// since been deallocated.
//
// In this case, the values in question hold (non-zero) unique-ids
// that zero themselves out when dropped, and are wrapped in another
// type with a destructor that asserts that the ids it references are
// indeed non-zero (i.e., effectively checking that the id's are not
// dropped while there are still any outstanding references).
//
// However, the values in question are also formed into a
// cyclic-structure, ensuring that there is no way for all of the
// conditions above to be satisfied, meaning that if the dropck is
// sound, it should reject this code.



use std::cell::Cell;
use id::Id;

mod s {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static S_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// generates globally unique count (global across the current
    /// process, that is)
    pub fn next_count() -> usize {
        S_COUNT.fetch_add(1, Ordering::SeqCst) + 1
    }
}

mod id {
    use crate::s;

    /// Id represents a globally unique identifier (global across the
    /// current process, that is). When dropped, it automatically
    /// clears its `count` field, but leaves `orig_count` untouched,
    /// so that if there are subsequent (erroneous) invocations of its
    /// method (which is unsound), we can observe it by seeing that
    /// the `count` is 0 while the `orig_count` is non-zero.
    #[derive(Debug)]
    pub struct Id {
        orig_count: usize,
        count: usize,
    }

    impl Id {
        /// Creates an `Id` with a globally unique count.
        pub fn new() -> Id {
            let c = s::next_count();
            println!("building Id {}", c);
            Id { orig_count: c, count: c }
        }
        /// returns the `count` of self; should be non-zero if
        /// everything is working.
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
struct C<'a> {
    id: Id,
    v: Vec<CheckId<Cell<Option<&'a C<'a>>>>>,
}

impl<'a> HasId for Cell<Option<&'a C<'a>>> {
    fn count(&self) -> usize {
        match self.get() {
            None => 1,
            Some(c) => c.id.count(),
        }
    }
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { id: Id::new(), v: Vec::new() }
    }
}

fn f() {
    let (mut c1, mut c2);
    c1 = C::new();
    c2 = C::new();

    c1.v.push(CheckId(Cell::new(None)));
    c2.v.push(CheckId(Cell::new(None)));
    c1.v[0].v.set(Some(&c2));
    //~^ ERROR `c2` does not live long enough
    c2.v[0].v.set(Some(&c1));
    //~^ ERROR `c1` does not live long enough
}

fn main() {
    f();
}
