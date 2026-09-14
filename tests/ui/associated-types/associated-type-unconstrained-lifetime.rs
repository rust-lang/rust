//! Regression test for <https://github.com/rust-lang/rust/issues/22886>.
//! Test that associated type's inner state cannot be observed past
//! borrow end via saved reference.
//!
//! This was possible as trait implementation with unconstrained lifetime
//! allowed to use any lifetime for associated type, which introduced
//! soundness holes.
//!
//! Fixed by prohibiting use of unconstrained lifetimes on associated types
//! in <https://github.com/rust-lang/rust/pull/24461>.
//!
//! Related <https://github.com/rust-lang/rust/issues/22077>.

fn crash_please() {
    let mut iter = Newtype(Some(Box::new(0)));
    let saved = iter.next().unwrap();
    println!("{}", saved);
    iter.0 = None;
    println!("{}", saved);
}

struct Newtype(Option<Box<usize>>);

impl<'a> Iterator for Newtype { //~ ERROR E0207
    type Item = &'a Box<usize>;

    fn next(&mut self) -> Option<&Box<usize>> {
        self.0.as_ref()
    }
}

fn main() { }
