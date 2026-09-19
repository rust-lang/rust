// Regression test for #144547.

//@ revisions: min mgca
//@[mgca] check-pass
#![cfg_attr(mgca, feature(min_generic_const_args, macroless_generic_const_args))]

trait UnderlyingImpl<const MAX_SIZE: usize> {
    type InfoType: LevelInfo;
    type SupportedArray<T>;
}

trait LevelInfo {
    #[cfg(mgca)]
    #[rustc_always_gca]
    const SUPPORTED_SLOTS: usize;

    #[cfg(not(mgca))]
    const SUPPORTED_SLOTS: usize;
}

struct Info;

impl LevelInfo for Info {
    #[cfg(mgca)]
    const SUPPORTED_SLOTS: usize = core::direct_const_arg!(1);

    #[cfg(not(mgca))]
    const SUPPORTED_SLOTS: usize = 1;
}

struct SomeImpl;

impl<const MAX_SIZE: usize> UnderlyingImpl<MAX_SIZE> for SomeImpl {
    type InfoType = Info;
    type SupportedArray<T> = [T; <Self::InfoType as LevelInfo>::SUPPORTED_SLOTS];
    //[min]~^ ERROR generic parameters
}

fn main() {}
