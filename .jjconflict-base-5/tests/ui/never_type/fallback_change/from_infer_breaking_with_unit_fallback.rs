// issue: rust-lang/rust#66757
//
// This is a *minimization* of the issue.
// Note that the original version with the `?` does not fail anymore even with fallback to unit,
// see `tests/ui/never_type/fallback_change/question_mark_from_never.rs`.
//
//@ check-pass

struct E;

impl From<!> for E {
    fn from(_: !) -> E {
        E
    }
}

#[allow(unreachable_code)]
fn foo(never: !) {
    <E as From<!>>::from(never); // Ok
    <E as From<_>>::from(never); // Should the inference fail?
}

fn main() {}
