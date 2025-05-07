//@ compile-flags: -Zthreads=16

// original issue: https://github.com/rust-lang/rust/issues/129112
// Previously, the "next" solver asserted that each successful solution is only obtained once.
// This test exhibits a repro that, with next-solver + -Zthreads, triggered that old assert.
// In the presence of multithreaded solving, it's possible to concurrently evaluate things twice,
// which leads to replacing already-solved solutions in the global solution cache!
// We assume this is fine if we check to make sure they are solved the same way each time.

// This test only nondeterministically fails but that's okay, as it will be rerun by CI many times,
// so it should almost always fail before anything is merged. As other thread tests already exist,
// we already face this difficulty, probably. If we need to fix this by reducing the error margin,
// we should improve compiletest.

#[derive(Clone, Eq)] //~ ERROR [E0277]
pub struct Struct<T>(T);

impl<T: Clone, U> PartialEq<U> for Struct<T>
where
    U: Into<Struct<T>> + Clone
{
    fn eq(&self, _other: &U) -> bool {
        todo!()
    }
}

fn main() {}
