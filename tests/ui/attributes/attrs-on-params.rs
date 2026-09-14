// This checks that incorrect params on function parameters are caught

fn function(#[inline] param: u32) {
    //~^ ERROR attribute cannot be used on
    //~| ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes
}

trait Test {
    fn meow(
        #[rustc_splat] a1: u32,
    );
    fn meow2(
        #[rustc_splat(invalid)] a4: u32,
    );
}

type Meow = fn(
    #[rustc_splat] a1: u32,
);

extern "Rust" {
    fn meow2(
        #[rustc_splat] a1: u32,
    );
}

fn main() {}
