// Exercise trivially unsatisfied bounds on free const items.
// Their interaction with the evaluation of the initializer is interesting.
//
//@ revisions: mentioned unmentioned

#![feature(generic_const_items)]

// FIXME(generic_const_items): Try to get rid of error "entering unreachable error", it's
// unnecessary and actually caused by MIR pass `ImpossiblePredicates` replacing the body with the
// terminator `Unreachable` due to the unsatisfied bound which is subsequently reached.
//
// NOTE(#142293): However, don't think about suppressing the evaluation of the initializer if the
// bounds are "impossible". That'd be a SemVer hazard since it could cause downstream to fail to
// compile if upstream added a new trait impl which is undesirable[^1].
// [^1]: Strictly speaking that's already possible due to the one-impl rule.

const UNUSABLE: () = () //~ ERROR entering unreachable code
where
    for<'_delay> String: Copy;

fn scope() {
    // Ensure that we successfully reject references of consts with trivially unsatisfied bounds.
    #[cfg(mentioned)]
    let _ = UNUSABLE; //[mentioned]~ ERROR the trait bound `String: Copy` is not satisfied
}

const _BAD: () = <() as Unimplemented>::CT
where
    for<'_delay> (): Unimplemented;

trait Unimplemented { const CT: (); }

fn main() {}
