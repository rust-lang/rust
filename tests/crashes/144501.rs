//@ known-bug: rust-lang/rust#144501
//@ needs-rustc-debug-assertions
enum E {
    S0 {
        s: String,
    },
    Bar = {
        let x = 1;
        3
    },
}

static C: E = E::S1 { u: 23 };
