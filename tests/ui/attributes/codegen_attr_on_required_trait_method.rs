#![deny(unused_attributes)]
#![feature(linkage)]
#![feature(fn_align)]

trait Test {
    #[cold]
    //~^ ERROR cannot be used on required trait methods [unused_attributes]
    //~| WARN previously accepted
    fn method1(&self);
    #[link_section = "__TEXT,__text"]
    //~^ ERROR cannot be used on required trait methods [unused_attributes]
    //~| WARN previously accepted
    fn method2(&self);
    #[linkage = "common"]
    //~^ ERROR `#[linkage]` attribute cannot be used on required trait methods
    fn method3(&self);
    #[track_caller]
    fn method4(&self);
    #[rustc_align(1)]
    fn method5(&self);
}

fn main() {}
