// This ensures we don't ICE in situations like rust-lang/rust#127299.

trait Qux {
    fn bar() -> i32;
}

pub struct Lint {
    pub desc: &'static dyn Qux,
    //~^ ERROR cannot be made into an object
}

static FOO: &Lint = &Lint { desc: "desc" };
//~^ ERROR cannot be shared between threads safely
//~| ERROR cannot be made into an object
//~| ERROR cannot be made into an object

fn main() {}
