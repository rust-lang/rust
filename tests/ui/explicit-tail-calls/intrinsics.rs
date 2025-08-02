//@ run-pass
#![feature(explicit_tail_calls, core_intrinsics)]
#![expect(incomplete_features, internal_features)]

fn trans((): ()) {
    // transmute is lowered in a mir pass
    unsafe { become std::mem::transmute(()) }
}

fn main() {
    trans(());
}

