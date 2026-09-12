//! Regression test for <https://github.com/rust-lang/rust/issues/4736>.
//! This used to ICE.

struct NonCopyable(());

fn main() {
    let z = NonCopyable{ p: () }; //~ ERROR struct `NonCopyable` has no field named `p`
}
