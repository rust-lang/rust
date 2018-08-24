#![deny(deprecated)]
#![feature(dropck_parametricity)]

struct Foo;

impl Drop for Foo {
    #[unsafe_destructor_blind_to_params]
    //~^ ERROR use of deprecated attribute `dropck_parametricity`
    fn drop(&mut self) {}
}

fn main() {}
