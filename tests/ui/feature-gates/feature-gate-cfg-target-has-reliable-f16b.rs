//@ ignore-s390x
//@ ignore-wasm
//@ compile-flags: --check-cfg=cfg(target_has_reliable_f16b)

fn main() {
    cfg!(target_has_reliable_f16b);
    //~^ ERROR `cfg(target_has_reliable_f16b)` is experimental and subject to change
}
