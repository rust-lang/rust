// This is testing an attempt to corrupt the discriminant of the match
// arm in a guard, followed by an attempt to continue matching on that
// corrupted discriminant in the remaining match arms.
//
// Basically this is testing that our new NLL feature of emitting a
// fake read on each match arm is catching cases like this.
//
// This case is interesting because a borrow of **x is untracked, because **x is
// immutable. However, for matches we care that **x refers to the same value
// until we have chosen a match arm.
#![feature(nll)]
struct ForceFnOnce;
fn main() {
    let mut x = &mut &Some(&2);
    let force_fn_once = ForceFnOnce;
    match **x {
        None => panic!("unreachable"),
        Some(&_) if {
            // ForceFnOnce needed to exploit #27282
            (|| { *x = &None; drop(force_fn_once); })();
            //~^ ERROR cannot mutably borrow `x` in match guard [E0510]
            false
        } => {}
        Some(&a) if { // this binds to garbage if we've corrupted discriminant
            println!("{}", a);
            panic!()
        } => {}
        _ => panic!("unreachable"),
    }
}
