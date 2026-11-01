//@ edition: 2021
//@ compile-flags: --crate-type=lib

// Regression test for https://github.com/rust-lang/rust/issues/150175.
//
// Moving a `&mut`-typed captured variable out of a closure (here through the
// implicit `into_iter` of a `for` loop) is an E0507. The "consider creating a
// fresh reborrow" suggestion rewrites the move site to `&mut *foos`, but that
// only compiles when the capture can be borrowed mutably there:
//
//   * an `Fn` closure holds its captures through `&self`, so `&mut *foos`
//     cannot be borrowed (E0596). The suggestion is `MachineApplicable`, so
//     offering it here produces code that does not compile -- it must be
//     suppressed (this is the bug).
//   * an `FnMut` closure holds its captures through `&mut self`, so the
//     reborrow is valid and the suggestion must still be offered.

pub struct Foo;

pub fn in_fn<F: Fn() -> Result<(), ()>>(f: F) -> Result<(), ()> {
    f()
}

pub fn in_fn_mut<F: FnMut() -> Result<(), ()>>(mut f: F) -> Result<(), ()> {
    f()
}

// `Fn` closure: a mutable reborrow is impossible, so no suggestion is offered.
pub fn fn_closure(foos: &mut [&mut Foo]) -> Result<(), ()> {
    in_fn(|| {
        for _ in foos {
            //~^ ERROR cannot move out of `foos`, a captured variable in an `Fn` closure
        }
        Ok(())
    })
}

// `FnMut` closure: a mutable reborrow is valid, so the suggestion is offered.
pub fn fn_mut_closure(foos: &mut [&mut Foo]) -> Result<(), ()> {
    in_fn_mut(|| {
        for _ in foos {
            //~^ ERROR cannot move out of `foos`, a captured variable in an `FnMut` closure
        }
        Ok(())
    })
}
