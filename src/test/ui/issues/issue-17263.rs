// This checks diagnostic quality for cases where AST-borrowck treated
// `Box<T>` as other types (see rust-lang/rfcs#130). NLL again treats
// `Box<T>` specially. We capture the differences via revisions.

// revisions: ast nll
//[ast]compile-flags: -Z borrowck=ast
//[nll]compile-flags: -Z borrowck=migrate -Z two-phase-borrows

// don't worry about the --compare-mode=nll on this test.
// ignore-compare-mode-nll
#![feature(box_syntax, rustc_attrs)]

struct Foo { a: isize, b: isize }
#[rustc_error] // rust-lang/rust#49855
fn main() { //[nll]~ ERROR compilation successful
    let mut x: Box<_> = box Foo { a: 1, b: 2 };
    let (a, b) = (&mut x.a, &mut x.b);
    //[ast]~^ ERROR cannot borrow `x` (via `x.b`) as mutable more than once at a time

    let mut foo: Box<_> = box Foo { a: 1, b: 2 };
    let (c, d) = (&mut foo.a, &foo.b);
    //[ast]~^ ERROR cannot borrow `foo` (via `foo.b`) as immutable

    // We explicitly use the references created above to illustrate
    // that NLL is accepting this code *not* because of artificially
    // short lifetimes, but rather because it understands that all the
    // references are of disjoint parts of memory.
    use_imm(d);
    use_mut(c);
    use_mut(b);
    use_mut(a);
}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
