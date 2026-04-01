//@ check-pass

// The doc comment here is ignored. This is a bug, but #95267 showed that
// existing programs rely on this behaviour, and changing it would require some
// care and a transition period.
macro_rules! f {
    (
        /// ab
    ) => {};
}

fn main() {
    f!();
}
