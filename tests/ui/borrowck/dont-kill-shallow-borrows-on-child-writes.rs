//! Regression test for <https://github.com/rust-lang/rust/issues/160599>. On assignment statements,
//! borrowck's dataflow analysis kills borrows that it knows conflict with the assigment: a borrow
//! of a place can't be live anymore after that place is assigned over. Previously, this didn't
//! account for fake borrows being shallow: it would kill any shallow borrows that would have
//! conflicted if they were normal borrows. This made it possible to circumvent fake borrows for
//! match guards and indexing expressions.

fn test_match_guard() {
    let mut a = (Some(&42u64), 0u8);
    let mut b = (None::<&u64>, 0u8);
    let mut p = &mut a;
    // Writing to `(*p).1` in the match guard previously killed the fake borrow of `p` in the guard,
    // making it possible to mutate `p` despite `(*p).0` being matched on. This would reach the
    // `Some(r)` branch with `(*p).0` being `None`, so the `r` binding was invalid.
    match p.0 {
        Some(_) if { p.1 = 1; p = &mut b; false } => unreachable!(),
        //~^ ERROR: cannot assign `p` in match guard
        Some(r) => println!("{r}"),
        None => unreachable!(),
    }
}

fn test_indexing() {
    let mut x: &mut [&mut [u32]] = &mut [&mut [0]];
    let y: &mut [&mut [u32]] = &mut [];
    // Writing to `x[0][0]` previously killed the fake borrow of `x` in the index expression, making
    // it possible to access `y[0]` without a bounds-check.
    x[0][{ x[0][0] = 1; x = y; 0 }];
    //~^ ERROR: cannot assign `x` in indexing expression
}

fn main() {}
