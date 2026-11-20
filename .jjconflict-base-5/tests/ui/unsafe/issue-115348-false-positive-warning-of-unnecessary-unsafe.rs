// Regression test for #115348.

unsafe fn uwu() {}

// Tests that the false-positive warning "unnecessary `unsafe` block"
// should not be reported, when the error "non-exhaustive patterns"
// appears.

fn foo(x: Option<u32>) {
    match x {
        //~^ ERROR non-exhaustive patterns: `None` not covered
        Some(_) => unsafe { uwu() },
    }
}

fn main() {}
