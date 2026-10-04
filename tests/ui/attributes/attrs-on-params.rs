// This checks that incorrect params on function parameters are caught

fn function(#[inline] param: u32) {
    //~^ ERROR attribute cannot be used on
    //~| ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes
}

trait Test {
    fn meow(
        #[rustc_splat] a1: u32,
        //~^ ERROR the `rustc_splat` attribute is an experimental feature
    );
    fn meow2(
        #[rustc_splat(invalid)] a4: u32,
        //~^ ERROR the `rustc_splat` attribute is an experimental feature
        //~| ERROR malformed `rustc_splat` attribute input
    );
}

type Meow = fn(
    #[rustc_splat] a1: u32,
    //~^ ERROR the `rustc_splat` attribute is an experimental feature
);

extern "Rust" {
    fn meow2(
        #[rustc_splat] a1: u32,
        //~^ ERROR the `rustc_splat` attribute is an experimental feature
    );
}

fn main() {}
