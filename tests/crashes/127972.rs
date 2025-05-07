//@ known-bug: #127962
#![feature(generic_const_exprs, const_arg_path)]

fn zero_init<const usize: usize>() -> Substs1<{ (N) }> {
    Substs1([0; { (usize) }])
}
