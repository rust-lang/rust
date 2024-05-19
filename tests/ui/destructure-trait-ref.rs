// The regression test for #15031 to make sure destructuring trait
// reference work properly.

#![feature(box_patterns)]

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
    let box x = Box::new(1isize) as Box<dyn T>;
    //~^ ERROR type `Box<dyn T>` cannot be dereferenced

    // n > m
    let &&x = &1isize as &dyn T;
    //~^ ERROR mismatched types
    //~| expected trait object `dyn T`
    //~| found reference `&_`
    let &&&x = &(&1isize as &dyn T);
    //~^ ERROR mismatched types
    //~| expected trait object `dyn T`
    //~| found reference `&_`
    let box box x = Box::new(1isize) as Box<dyn T>;
    //~^ ERROR mismatched types
    //~| expected trait object `dyn T`
    //~| found struct `Box<_>`
}
