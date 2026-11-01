// This test checks that there is no ICE with this code
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags:--check-cfg=cfg()

fn main() {
    #[cfg(crossbeam_loom)]
    //~^ WARNING unexpected `cfg` condition name
    {}
}
