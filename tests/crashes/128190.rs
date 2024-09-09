//@ known-bug: rust-lang/rust#128190

fn a(&self) {
    15
}

reuse a as b {  struct S; }
