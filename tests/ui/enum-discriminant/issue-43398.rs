// run-pass

#![feature(core_intrinsics)]
#![feature(repr128)]
//~^ WARN the feature `repr128` is incomplete

#[repr(i128)]
enum Big { A, B }

fn main() {
    println!("{} {:?}",
        std::intrinsics::discriminant_value(&Big::A),
        std::mem::discriminant(&Big::B));
}
