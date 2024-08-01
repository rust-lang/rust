//@ known-bug: #127962
#![feature(generic_const_exprs)]

fn zero_init<const usize: usize>() -> Substs1<{ (N) }> {
    Substs1([0; { (usize) }])
}
