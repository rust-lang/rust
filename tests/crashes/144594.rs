//@ known-bug: rust-lang/rust#144594
reuse a as b {
    || {
        use std::ops::Add;
        x.add
    }
}
