// Syntactically, a `const` item inside an `extern { ... }` block is not allowed.

fn main() {}

#[cfg(false)]
extern "C" {
    const A: isize; //~ ERROR extern items cannot be `const`
    const B: isize = 42; //~ ERROR extern items cannot be `const`
}
