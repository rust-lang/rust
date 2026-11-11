// gate-test-link_enzyme_intrinsics

unsafe extern "C" {
    fn __enzyme_autodiff();
    //~^ ERROR linking to Enzyme intrinsics is experimental

    fn __enzyme_fwddiff();
    //~^ ERROR linking to Enzyme intrinsics is experimental

    fn __enzyme_augmentfwd();
    //~^ ERROR linking to Enzyme intrinsics is experimental

    fn __enzyme_reverse();
    //~^ ERROR linking to Enzyme intrinsics is experimental

    #[link_name = "__enzyme_autodiff"]
    fn autodiff();
    //~^ ERROR linking to Enzyme intrinsics is experimental

    static __enzyme_dup: i32;
    //~^ ERROR linking to Enzyme intrinsics is experimental

    #[link_name = "__enzyme_dup"]
    static ENZYME_DUP: i32;
    //~^ ERROR linking to Enzyme intrinsics is experimental

    static enzyme_dup: i32;
}

fn main() {}
