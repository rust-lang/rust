// This test checks that there is no ICE with this code
//
//@ check-pass
//@ compile-flags:--check-cfg=cfg() -Z unstable-options

fn main() {
    #[cfg(crossbeam_loom)]
    //~^ WARNING unexpected `cfg` condition name
    {}
}
