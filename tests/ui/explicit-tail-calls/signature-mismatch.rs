#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
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


fn main() {}
