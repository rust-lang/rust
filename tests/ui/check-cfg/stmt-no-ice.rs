// This test checks that there is no ICE with this code
//
//@ check-pass
//@ compile-flags:--check-cfg=cfg()

fn main() {
    #[cfg(crossbeam_loom)]
    //~^ WARNING unexpected `cfg` condition name
    {}
}
