//@ run-pass
#![feature(explicit_tail_calls, core_intrinsics)]
#![expect(incomplete_features, internal_features)]

fn trans((): ()) {
    // transmute is lowered in a mir pass
    unsafe { become std::mem::transmute(()) }
}

fn cats(x: u64) -> u32 {
    become std::intrinsics::ctlz(x)
}

fn main() {
    trans(());
    assert_eq!(cats(17), 59);
}

