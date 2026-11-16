// This checks that incorrect params on function parameters are caught

fn function(#[inline] param: u32) {
    //~^ ERROR attribute cannot be used on
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
