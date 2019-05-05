#![feature(nll)]
#![feature(bind_by_move_pattern_guards)]

enum VecWrapper { A(Vec<i32>) }

fn foo(x: VecWrapper) -> usize {
    match x {
        VecWrapper::A(v) if { drop(v); false } => 1,
        //~^ ERROR cannot move out of `v` in pattern guard
        VecWrapper::A(v) => v.len()
    }
}

fn main() {
    foo(VecWrapper::A(vec![107]));
}
