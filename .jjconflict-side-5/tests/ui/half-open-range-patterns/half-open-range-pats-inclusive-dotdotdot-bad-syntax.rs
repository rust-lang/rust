// Test that `...X` range-to patterns are syntactically invalid.
//
// See https://github.com/rust-lang/rust/pull/67258#issuecomment-565656155
// for the reason why. To summarize, we might want to introduce `...expr` as
// an expression form for splatting (or "untupling") in an expression context.
// While there is no syntactic ambiguity with `...X` in a pattern context,
// there's a potential confusion factor here, and we would prefer to keep patterns
// and expressions in-sync. As such, we do not allow `...X` in patterns either.

fn main() {}

#[cfg(false)]
fn syntax() {
    match scrutinee {
        ...X => {} //~ ERROR range-to patterns with `...` are not allowed
        ...0 => {} //~ ERROR range-to patterns with `...` are not allowed
        ...'a' => {} //~ ERROR range-to patterns with `...` are not allowed
        ...0.0f32 => {} //~ ERROR range-to patterns with `...` are not allowed
    }
}

fn syntax2() {
    macro_rules! mac {
        ($e:expr) => {
            let ...$e; //~ ERROR range-to patterns with `...` are not allowed
            //~^ ERROR refutable pattern in local binding
        }
    }

    mac!(0);
}
