//! Regression test for issue <https://github.com/rust-lang/rust/issues/156713>. In the `fails`
//! case, a drop used to be emitted for the temporary holding the operand of the tuple constructor
//! after it's moved into the tuple. The borrow-checker treated the unwind edge from that drop as
//! live, as if it could panic. Were execution to unwind from there, the match scrutinee tuple would
//! be dropped after the `foo` variable, due to pre-2024 editions lacking a temporary scope around
//! block tail expressions. Since the `Bar` in the tuple may access its borrow of `foo` in its
//! destructor, it was an error for the tuple to be dropped after `foo` on the unwind path.
//!
//! As of PR <https://github.com/rust-lang/rust/pull/158281>, a drop is no longer emitted for the
//! tuple operand after it's been moved, so the problematic unwind path doesn't exist anymore.
//!
//! The same drop used to be emitted in the `works` case too, but the drop for the match scrutinee
//! temporary was missing from the unwind path from the tuple operand's drop due to issue
//! <https://github.com/rust-lang/rust/issues/47949>, so there was no drop order conflict.
//@ edition: 2015
//@ check-pass

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {}
}

struct Bar<'a>(&'a Foo);
impl Drop for Bar<'_> {
    fn drop(&mut self) {}
}

// This compiles
fn works() {
    let foo = Foo;
    let bar = Bar(&foo);
    drop(match { (bar,) } {
        args => args,
    })
}

// This used to error
fn fails() {
    let foo = Foo;
    let bar = Bar(&foo);
    drop(match (bar,) {
        args => args,
    })
}

fn main() {}
