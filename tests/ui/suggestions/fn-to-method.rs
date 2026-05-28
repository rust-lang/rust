//@ revisions: normal import_trait_associated_functions
#![cfg_attr(import_trait_associated_functions, feature(import_trait_associated_functions))]
struct Foo;
//[import_trait_associated_functions]~^ HELP consider importing one of these associated functions
//[import_trait_associated_functions]~| HELP consider importing this associated function

impl Foo {
    fn bar(self) {}
}

fn main() {
    let x = cmp(&1, &2);
    //~^ ERROR cannot find function `cmp` in this scope
    //[normal]~| HELP use the `.` operator to call the method `Ord::cmp` on `&{integer}`

    let y = len([1, 2, 3]);
    //~^ ERROR cannot find function `len` in this scope
    //[normal]~| HELP use the `.` operator to call the method `len` on `&[{integer}]`

    let z = bar(Foo);
    //~^ ERROR cannot find function `bar` in this scope
    //~| HELP use the `.` operator to call the method `bar` on `Foo`
}
