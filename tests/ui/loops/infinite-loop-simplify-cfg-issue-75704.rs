// Caused an infinite loop during SimlifyCfg MIR transform previously.
//
//@ build-pass

fn main() {
    loop { continue; }
}

// https://github.com/rust-lang/rust/issues/75704
