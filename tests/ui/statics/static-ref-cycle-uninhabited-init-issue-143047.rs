// Regression test for https://github.com/rust-lang/rust/issues/143047

enum Never {}

static X: &Never = weird(&X);
//~^ ERROR: encountered static that tried to access itself during initialization

const fn weird(a: &&Never) -> &'static Never {
    // SAFETY: our argument type has an unsatisfiable
    // library invariant; therefore, this code is unreachable.
    unsafe { std::hint::unreachable_unchecked() };
}

fn main() {}
