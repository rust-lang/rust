#![feature(box_syntax)]

fn needs_fn<F>(x: F) where F: Fn(isize) -> isize {}

fn main() {
    let _: () = (box |_: isize| {}) as Box<dyn FnOnce(isize)>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `std::boxed::Box<dyn std::ops::FnOnce(isize)>`
    let _: () = (box |_: isize, isize| {}) as Box<dyn Fn(isize, isize)>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `std::boxed::Box<dyn std::ops::Fn(isize, isize)>`
    let _: () = (box || -> isize { unimplemented!() }) as Box<dyn FnMut() -> isize>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `std::boxed::Box<dyn std::ops::FnMut() -> isize>`

    needs_fn(1);
    //~^ ERROR expected a `std::ops::Fn<(isize,)>` closure, found `{integer}`
}
