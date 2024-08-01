//@ known-bug: rust-lang/rust#127299
trait Qux {
    fn bar() -> i32;
}

pub struct Lint {
    pub desc: &'static Qux,
}

static FOO: &Lint = &Lint { desc: "desc" };

fn main() {}
