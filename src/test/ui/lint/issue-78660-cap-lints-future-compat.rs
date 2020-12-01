// compile-flags: -D warnings --cap-lints allow
// check-pass

// Regression test for issue #78660
// Tests that we don't ICE when a future-incompat-report lint has
// has a command-line source, but is capped to allow

fn main() {
    ["hi"].into_iter();
}
