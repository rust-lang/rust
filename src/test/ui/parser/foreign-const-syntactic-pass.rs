// Syntactically, a `const` item inside an `extern { ... }` block is allowed.

// check-pass

fn main() {}

#[cfg(FALSE)]
extern {
    const A: isize;
    const B: isize = 42;
}
