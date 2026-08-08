//@ known-bug: rust-lang/rust#146084
struct S<const N: [()] = { loop {} }>;
