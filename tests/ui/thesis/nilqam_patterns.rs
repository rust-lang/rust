//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_STATS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1
//! Patterns suggested by Nico, lqd, and Amanda

#![allow(dropping_references)]

// #[warn(clippy::borrow_pats)]
mod part_return {
    struct List<T> {
        value: T,
        next: Option<Box<List<T>>>,
    }

    fn next_unchecked<T>(x: &List<T>) -> &List<T> {
        let Some(n) = &x.next else {
            panic!();
        };
        &*n
    }
}

// #[warn(clippy::borrow_pats)]
mod overwrite_in_loop {
    trait SomethingMut {
        fn something(&mut self);
    }
    struct List<T> {
        value: T,
        next: Option<Box<List<T>>>,
    }

    fn last<T: SomethingMut>(x: &mut List<T>) -> &List<T> {
        let mut p = x;
        loop {
            p.value.something();
            if p.next.is_none() {
                break p;
            }
            p = p.next.as_mut().unwrap();
        }
    }
}

/// Such loan kills should be detected by named references.
/// AFAIK, they can only occur for named references. Tracking them from the
/// owned value could result to incorrect counting, as a reference can have
/// multiple brockers.
#[warn(clippy::borrow_pats)]
fn mut_named_ref_non_kill() {
    let mut x = 1;
    let mut y = 1;
    let mut p: &u32 = &x;
    // x is borrowed here
    drop(p);

    // variable p is dead here
    p = &y;
    // x is not borrowed here
    drop(p);
}

// #[forbid(clippy::borrow_pats)]
mod early_kill_for_non_drop {
    // This wouldn't work if `X` would implement `Drop`
    // Credit: https://github.com/rust-lang/rfcs/pull/2094#issuecomment-320171945
    struct X<'a> {
        val: &'a usize,
    }
    fn mutation_works_because_non_drop() {
        let mut x = 42;
        let y = X { val: &x };
        x = 50;
    }
}

// FIXME: Maybe: https://github.com/nikomatsakis/nll-rfc/issues/38
// FIXME: Maybe include recursion?
fn main() {}
