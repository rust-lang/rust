//@ compile-flags: --check-cfg=cfg(target_has_reliable_f16,target_has_reliable_f16_math,target_has_reliable_f128,target_has_reliable_f128_math)

fn main() {
    cfg!(target_has_reliable_f16);
    //~^ ERROR `cfg(target_has_reliable_f16)` is experimental and subject to change
    cfg!(target_has_reliable_f16_math);
    //~^ ERROR `cfg(target_has_reliable_f16_math)` is experimental and subject to change
    cfg!(target_has_reliable_f128);
    //~^ ERROR `cfg(target_has_reliable_f128)` is experimental and subject to change
    cfg!(target_has_reliable_f128_math);
    //~^ ERROR `cfg(target_has_reliable_f128_math)` is experimental and subject to change
}
