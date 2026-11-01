//@ check-pass
//! Test that `GatherLocalsVisitor` only visits expressions in guard patterns when checking the
//! expressions, and not a second time when visiting the pattern. If locals are declared inside the
//! the guard expression, it would ICE if visited twice ("evaluated expression more than once").

#![feature(guard_patterns)]
#![expect(incomplete_features)]

fn main() {
    match (0,) {
        // FIXME(guard_patterns): liveness lints don't work yet; this will ICE without the `_`.
        (_ if { let _x = false; _x },) => {}
        _ => {}
    }
}
