// xfail-fast Does not work with main in a submodule

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
mod m {
    #[legacy_exports];
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
}

#[cfg(target_os = "win32")]
mod m {
    #[legacy_exports];
    #[cfg(target_arch = "x86")]
    fn main() {
        assert rusti::pref_align_of::<u64>() == 8u;
        assert rusti::min_align_of::<u64>() == 8u;
    }
}
