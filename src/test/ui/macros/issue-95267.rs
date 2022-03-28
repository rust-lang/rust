// check-pass

// This is a valid macro. Commit 4 in #95159 broke things such that it failed
// with a "missing tokens in macro arguments" error, as reported in #95267.
macro_rules! f {
    (
        /// ab
    ) => {};
}

fn main() {
    f!();
}
