// run-pass

#![feature(core_intrinsics)]
#![feature(repr128)]

#[repr(i128)]
enum Big { A, B }

fn main() {
    unsafe {
        println!("{} {:?}",
            std::intrinsics::discriminant_value(&Big::A),
            std::mem::discriminant(&Big::B));
    }
}
