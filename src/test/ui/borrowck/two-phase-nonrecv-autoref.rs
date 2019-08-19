// revisions: nll
//[nll]compile-flags: -Z borrowck=mir

//[g2p]compile-flags: -Z borrowck=mir -Z two-phase-beyond-autoref
// the above revision is disabled until two-phase-beyond-autoref support is better

// This is a test checking that when we limit two-phase borrows to
// method receivers, we do not let other kinds of auto-ref to leak
// through.
//
// The g2p revision illustrates the "undesirable" behavior you would
// otherwise observe without limiting the phasing to autoref on method
// receivers (namely, in many cases demonstrated below, the error
// would not arise).

// (If we revise the compiler or this test so that the g2p revision
// passes, turn the `rustc_attrs` feature back on and tag the `fn
// main` with `#[rustc_error]` so that this remains a valid
// compile-fail test.)
//
// #![feature(rustc_attrs)]

use std::ops::{Index, IndexMut};

fn foo(x: &mut u32, y: u32) {
    *x += y;
}

fn deref_coercion(x: &mut u32) {
    foo(x, *x);
    // Above error is a known limitation of AST borrowck
}

// While adding a flag to adjustments (indicating whether they
// should support two-phase borrows, here are the cases I
// encountered:
//
// - [x] Resolving overloaded_call_traits (call, call_mut, call_once)
// - [x] deref_coercion (shown above)
// - [x] coerce_unsized e.g., `&[T; n]`, `&mut [T; n] -> &[T]`,
//                      `&mut [T; n] -> &mut [T]`, `&Concrete -> &Trait`
// - [x] Method Call Receivers (the case we want to support!)
// - [x] ExprKind::Index and ExprKind::Unary Deref; only need to handle coerce_index_op
// - [x] overloaded_binops

fn overloaded_call_traits() {
    // Regarding overloaded call traits, note that there is no
    // scenario where adding two-phase borrows should "fix" these
    // cases, because either we will resolve both invocations to
    // `call_mut` (in which case the inner call requires a mutable
    // borrow which will conflict with the outer reservation), or we
    // will resolve both to `call` (which will just work, regardless
    // of two-phase borrow support), or we will resolve both to
    // `call_once` (in which case the inner call requires moving the
    // receiver, invalidating the outer call).

    fn twice_ten_sm<F: FnMut(i32) -> i32>(f: &mut F) {
        f(f(10));
        //[nll]~^   ERROR cannot borrow `*f` as mutable more than once at a time
        //[g2p]~^^ ERROR cannot borrow `*f` as mutable more than once at a time
    }
    fn twice_ten_si<F: Fn(i32) -> i32>(f: &mut F) {
        f(f(10));
    }
    fn twice_ten_so<F: FnOnce(i32) -> i32>(f: Box<F>) {
        f(f(10));
        //[nll]~^   ERROR use of moved value: `f`
        //[g2p]~^^  ERROR use of moved value: `f`
    }

    fn twice_ten_om(f: &mut dyn FnMut(i32) -> i32) {
        f(f(10));
        //[nll]~^   ERROR cannot borrow `*f` as mutable more than once at a time
        //[g2p]~^^  ERROR cannot borrow `*f` as mutable more than once at a time
    }
    fn twice_ten_oi(f: &mut dyn Fn(i32) -> i32) {
        f(f(10));
    }
    fn twice_ten_oo(f: Box<dyn FnOnce(i32) -> i32>) {
        f(f(10));
        //[nll]~^   ERROR use of moved value: `f`
        //[g2p]~^^  ERROR use of moved value: `f`
    }

    twice_ten_sm(&mut |x| x + 1);
    twice_ten_si(&mut |x| x + 1);
    twice_ten_so(Box::new(|x| x + 1));
    twice_ten_om(&mut |x| x + 1);
    twice_ten_oi(&mut |x| x + 1);
    twice_ten_oo(Box::new(|x| x + 1));
}

trait TwoMethods {
    fn m(&mut self, x: i32) -> i32 { x + 1 }
    fn i(&self, x: i32) -> i32 { x + 1 }
}

struct T;

impl TwoMethods for T { }

struct S;

impl S {
    fn m(&mut self, x: i32) -> i32 { x + 1 }
    fn i(&self, x: i32) -> i32 { x + 1 }
}

impl TwoMethods for [i32; 3] { }

fn double_access<X: Copy>(m: &mut [X], s: &[X]) {
    m[0] = s[1];
}

fn coerce_unsized() {
    let mut a = [1, 2, 3];

    // This is not okay.
    double_access(&mut a, &a);
    //[nll]~^   ERROR cannot borrow `a` as immutable because it is also borrowed as mutable [E0502]
    //[g2p]~^^  ERROR cannot borrow `a` as immutable because it is also borrowed as mutable [E0502]

    // But this is okay.
    a.m(a.i(10));
    // Above error is an expected limitation of AST borrowck
}

struct I(i32);

impl Index<i32> for I {
    type Output = i32;
    fn index(&self, _: i32) -> &i32 {
        &self.0
    }
}

impl IndexMut<i32> for I {
    fn index_mut(&mut self, _: i32) -> &mut i32 {
        &mut self.0
    }
}

fn coerce_index_op() {
    let mut i = I(10);
    i[i[3]] = 4;
    //[nll]~^  ERROR cannot borrow `i` as immutable because it is also borrowed as mutable [E0502]

    i[3] = i[4];

    i[i[3]] = i[4];
    //[nll]~^  ERROR cannot borrow `i` as immutable because it is also borrowed as mutable [E0502]
}

fn main() {

    // As a reminder, this is the basic case we want to ensure we handle.
    let mut v = vec![1, 2, 3];
    v.push(v.len());
    // Error above is an expected limitation of AST borrowck

    // (as a rule, pnkfelix does not like to write tests with dead code.)

    deref_coercion(&mut 5);
    overloaded_call_traits();


    let mut s = S;
    s.m(s.i(10));
    // Error above is an expected limitation of AST borrowck

    let mut t = T;
    t.m(t.i(10));
    // Error above is an expected limitation of AST borrowck

    coerce_unsized();
    coerce_index_op();
}
