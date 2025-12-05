//@ run-pass
//@ compile-flags: --cfg fooA --cfg fooB --check-cfg=cfg(fooA,fooB,fooC,bar)

// fooA AND !bar
#[cfg(all(fooA, not(bar)))]
fn foo1() -> isize { 1 }

// !fooA AND !bar
#[cfg(all(not(fooA), not(bar)))]
fn foo2() -> isize { 2 }

// fooC OR (fooB AND !bar)
#[cfg(any(fooC, all(fooB, not(bar))))]
fn foo2() -> isize { 3 }

// fooA AND bar
#[cfg(all(fooA, bar))]
fn foo3() -> isize { 2 }

// !(fooA AND bar)
#[cfg(not(all(fooA, bar)))]
fn foo3() -> isize { 3 }

pub fn main() {
    assert_eq!(1, foo1());
    assert_eq!(3, foo2());
    assert_eq!(3, foo3());
}
