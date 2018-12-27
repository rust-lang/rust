// This is testing an attempt to corrupt the discriminant of the match
// arm in a guard, followed by an attempt to continue matching on that
// corrupted discriminant in the remaining match arms.
//
// Basically this is testing that our new NLL feature of emitting a
// fake read on each match arm is catching cases like this.
//
// This case is interesting because it includes a guard that
// diverges, and therefore a single final fake-read at the very end
// after the final match arm would not suffice.
//
// It is also interesting because the access to the corrupted data
// occurs in the pattern-match itself, and not in the guard
// expression.

#![feature(nll)]

struct ForceFnOnce;

fn main() {
    let mut x = &mut Some(&2);
    let force_fn_once = ForceFnOnce;
    match x {
        &mut None => panic!("unreachable"),
        &mut Some(&_)
            if {
                // ForceFnOnce needed to exploit #27282
                (|| { *x = None; drop(force_fn_once); })();
                //~^ ERROR cannot mutably borrow `x` in match guard [E0510]
                false
            } => {}

        // this segfaults if we corrupted the discriminant, because
        // the compiler gets to *assume* that it cannot be the `None`
        // case, even though that was the effect of the guard.
        &mut Some(&2)
            if {
                panic!()
            } => {}
        _ => panic!("unreachable"),
    }
}
