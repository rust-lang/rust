#[repr(C)]
struct ReallyBig {
    _a: [u8; usize::MAX],
}

// The limit for "too big for the current architecture" is dependent on the target pointer size
// but is artificially limited due to LLVM's internal architecture
// logic based on rustc_target::abi::TargetDataLayout::obj_size_bound()
const fn max_size() -> usize {
    if usize::BITS < 61 {
        1 << (usize::BITS - 1)
    } else {
        1 << 61
    }
}

extern "C" {
    static FOO: [u8; 1];
    static BAR: [u8; max_size() - 1];
    static BAZ: [u8; max_size()]; //~ ERROR extern static is too large
    static UWU: [usize; usize::MAX]; //~ ERROR extern static is too large
    static A: ReallyBig; //~ ERROR extern static is too large
}

fn main() {}
