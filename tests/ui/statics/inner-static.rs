//! Test for inner statics with the same name.
//!
//! Before, the path name for all items defined in methods of traits and impls never
//! took into account the name of the method. This meant that if you had two statics
//! of the same name in two different methods the statics would end up having the
//! same symbol named (even after mangling) because the path components leading to
//! the symbol were exactly the same (just __extensions__ and the static name).
//!
//! It turns out that if you add the symbol "A" twice to LLVM, it automatically
//! makes the second one "A1" instead of "A". What this meant is that in local crate
//! compilations we never found this bug. Even across crates, this was never a
//! problem. The problem arises when you have generic methods that don't get
//! generated at compile-time of a library. If the statics were re-added to LLVM by
//! a client crate of a library in a different order, you would reference different
//! constants (the integer suffixes wouldn't be guaranteed to be the same).

//@ run-pass
//@ aux-build:inner_static.rs


extern crate inner_static;

pub fn main() {
    let a = inner_static::A::<()> { v: () };
    let b = inner_static::B::<()> { v: () };
    let c = inner_static::test::A::<()> { v: () };
    assert_eq!(a.bar(), 2);
    assert_eq!(b.bar(), 4);
    assert_eq!(c.bar(), 6);
}
