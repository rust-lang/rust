// Issue 27282: This is a variation on issue-27282-move-ref-mut-into-guard.rs
//
// It reborrows instead of moving the `ref mut` pattern borrow. This
// means that our conservative check for mutation in guards will
// reject it. But I want to make sure that we continue to reject it
// (under NLL) even when that conservaive check goes away.


#![feature(bind_by_move_pattern_guards)]
#![feature(nll)]

fn main() {
    let mut b = &mut true;
    match b {
        &mut false => {},
        ref mut r if { (|| { let bar = &mut *r; **bar = false; })();
        //~^ ERROR cannot borrow `r` as mutable, as it is immutable for the pattern guard
                             false } => { &mut *r; },
        &mut true => { println!("You might think we should get here"); },
        _ => panic!("surely we could never get here, since rustc warns it is unreachable."),
    }
}
