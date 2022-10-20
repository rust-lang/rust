#![feature(effects)]
#![feature(core_intrinsics)]

// check-pass

const fn foo() -> usize {
    std::intrinsics::size_of::<i32>()
}

fn main() {}
