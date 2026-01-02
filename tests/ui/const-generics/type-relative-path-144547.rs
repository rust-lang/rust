// Regression test for #144547.

//@ revisions: min mgca
//@[mgca] check-pass

#![cfg_attr(mgca, feature(min_generic_const_args))]
#![cfg_attr(mgca, expect(incomplete_features))]

trait UnderlyingImpl<const MAX_SIZE: usize> {
    type InfoType: LevelInfo;
    type SupportedArray<T>;
}

// FIXME: cfg_attr(..., type_const) is broken (search for sym::type_const in compiler/)
trait LevelInfo {
    #[cfg(mgca)]
    #[type_const]
    const SUPPORTED_SLOTS: usize;

    #[cfg(not(mgca))]
    const SUPPORTED_SLOTS: usize;
}

struct Info;

impl LevelInfo for Info {
    #[cfg(mgca)]
    #[type_const]
    const SUPPORTED_SLOTS: usize = 1;

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
