#![feature(core_intrinsics)]

pub fn not_instrprof_increment(_hash: u64, _num_counters: u32, _index: u32) {
}

fn main() {
    // COMPARE THIS WITH INTRINSIC INSERTION
    //not_instrprof_increment(1234 as u64, 314 as u32, 31 as u32);

    unsafe { core::intrinsics::instrprof_increment(1234 as u64, 314 as u32, 31 as u32) };
}