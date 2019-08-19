// The regression test for #15031 to make sure destructuring trait
// reference work properly.

#![feature(box_patterns)]
#![feature(box_syntax)]

trait T { fn foo(&self) {} }
impl T for isize {}

fn main() {
    // For an expression of the form:
    //
    //      let &...&x = &..&SomeTrait;
    //
    // Say we have n `&` at the left hand and m `&` right hand, then:
    // if n < m, we are golden;
    // if n == m, it's a derefing non-derefable type error;
    // if n > m, it's a type mismatch error.

    // n < m
    let &x = &(&1isize as &dyn T);
    let &x = &&(&1isize as &dyn T);
    let &&x = &&(&1isize as &dyn T);

    // n == m
    let &x = &1isize as &dyn T;      //~ ERROR type `&dyn T` cannot be dereferenced
    let &&x = &(&1isize as &dyn T);  //~ ERROR type `&dyn T` cannot be dereferenced
    let box x = box 1isize as Box<dyn T>;
    //~^ ERROR type `std::boxed::Box<dyn T>` cannot be dereferenced

    // n > m
    let &&x = &1isize as &dyn T;
    //~^ ERROR mismatched types
    //~| expected type `dyn T`
    //~| found type `&_`
    //~| expected trait T, found reference
    let &&&x = &(&1isize as &dyn T);
    //~^ ERROR mismatched types
    //~| expected type `dyn T`
    //~| found type `&_`
    //~| expected trait T, found reference
    let box box x = box 1isize as Box<dyn T>;
    //~^ ERROR mismatched types
    //~| expected type `dyn T`
    //~| found type `std::boxed::Box<_>`
}
