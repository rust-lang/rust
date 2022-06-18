#[repr(C)]
struct ReallyBig {
    _a: [u8; usize::MAX],
}

// The limit for "too big for the current architecture" is dependent on the target pointer size
// however it's artificially limited on 64 bits
// logic copied from rustc_target::abi::TargetDataLayout::obj_size_bound()
const fn max_size() -> usize {
    #[cfg(target_pointer_width = "16")]
    {
        1 << 15
    }

    #[cfg(target_pointer_width = "32")]
    {
        1 << 31
    }

    #[cfg(target_pointer_width = "64")]
    {
        1 << 47
    }

    #[cfg(not(any(
        target_pointer_width = "16",
        target_pointer_width = "32",
        target_pointer_width = "64"
    )))]
    {
        isize::MAX as usize
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
