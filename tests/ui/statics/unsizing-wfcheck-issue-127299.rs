// This ensures we don't ICE in situations like rust-lang/rust#127299.

trait Qux {
    fn bar() -> i32;
}

pub struct Lint {
    pub desc: &'static dyn Qux,
    //~^ ERROR is not dyn compatible
}

static FOO: &Lint = &Lint { desc: "desc" };
//~^ ERROR cannot be shared between threads safely
//~| ERROR is not dyn compatible

fn main() {}
