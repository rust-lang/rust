//@ run-pass
// This is a name resolution smoke test that ensures paths with more than one
// segment (e.g., `foo::bar`) resolve correctly.
// It also serves as a basic visibility test — confirming that a `pub` item
// inside a private module can still be accessed from outside that module.

mod foo {
    pub fn bar(_offset: usize) {}
}

fn main() { foo::bar(0); }
