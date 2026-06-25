#![expect(incomplete_features)]
#![feature(explicit_tail_calls, rust_tail_cc)]
#![feature(c_variadic)]

fn _f0((): ()) {
    become _g0(); //~ error: mismatched signatures
}

fn _g0() {}

fn _f1() {
    become _g1(()); //~ error: mismatched signatures
}

fn _g1((): ()) {}

extern "C" fn _f2() {
    become _g2(); //~ error: mismatched function ABIs
}

fn _g2() {}

fn _f3() {
    become _g3(); //~ error: mismatched function ABIs
}

extern "C" fn _g3() {}

extern "tail" fn _tailcc() {}

fn _f4() {
    // tailcc does not need the signatures to match,
    // but only tailcc can tail call tailcc.
    become _tailcc(); //~ error: mismatched function ABIs
}

fn main() {}
