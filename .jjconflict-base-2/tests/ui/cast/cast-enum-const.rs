//! regression test for <https://github.com/rust-lang/rust/issues/2428>
//@ run-pass

fn main() {
    const QUUX: isize = 5;

    enum Stuff {
        Bar = QUUX,
    }

    assert_eq!(Stuff::Bar as isize, QUUX);
}
