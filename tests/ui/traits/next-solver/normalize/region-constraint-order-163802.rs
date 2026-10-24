//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Two orderings of the same logical obligations. A regression in the next
// solver can make one ordering overflow or produce an ambiguity unexpectedly.
// This is a compile-pass regression, NOT a byte-for-byte canonical-response test.
#![allow(dead_code)]

trait Associated {
    type Output<'a>
    where
        Self: 'a;
}

trait Bound<'a> {}

fn require_bound<'a, T: Bound<'a>>() {}

fn forward<'a, T: 'a>()
where
    T: Associated<Output<'a> = T> + Bound<'a>,
    <T as Associated>::Output<'a>: Bound<'a>,
{
    require_bound::<'a, T>();
}

fn reversed<'a, T: 'a>()
where
    <T as Associated>::Output<'a>: Bound<'a>,
    T: Bound<'a> + Associated<Output<'a> = T>,
{
    require_bound::<'a, T>();
}

fn main() {}
