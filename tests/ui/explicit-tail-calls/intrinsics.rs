#![feature(explicit_tail_calls, core_intrinsics)]
#![expect(incomplete_features, internal_features)]

fn trans((): ()) {
    unsafe { become std::mem::transmute(()) } //~ error: tail calling intrinsics is not allowed

}

fn cats(x: u64) -> u32 {
    become std::intrinsics::ctlz(x) //~ error: tail calling intrinsics is not allowed
}

fn main() {}
