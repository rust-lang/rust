// xfail-win32 need to investigate alignment on windows

#[abi = "rust-intrinsic"]
native mod rusti {
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

#[cfg(target_arch = "x86")]
fn main() {
    assert rusti::pref_align_of::<u64>() == 8u;
    assert rusti::min_align_of::<u64>() == 4u;
}

#[cfg(target_arch = "x86_64")]
fn main() {
    assert rusti::pref_align_of::<u64>() == 8u;
    assert rusti::min_align_of::<u64>() == 8u;
}
