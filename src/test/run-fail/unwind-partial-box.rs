// error-pattern:fail

fn f() -> [int] { fail; }

// Voodoo. In unwind-alt we had to do this to trigger the bug. Might
// have been to do with memory allocation patterns.
fn prime() {
    @0;
}

fn partial() {
    let x = @f();
}

fn main() {
    prime();
    partial();
}