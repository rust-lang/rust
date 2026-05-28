//@ edition:2018
// Check that lifetime resolver enforces the lifetime name scoping
// rules correctly in various scenarios.

struct Foo<'a> {
    x: &'a isize
}

impl<'a> Foo<'a> {
    // &'a is inherited:
    fn m1(&self, arg: &'a isize) { }
    fn m2(&'a self) { }
    fn m3(&self, arg: Foo<'a>) { }

    // &'b is not:
    fn m4(&self, arg: &'b isize) { } //~ ERROR undeclared lifetime
    fn m5(&'b self) { } //~ ERROR undeclared lifetime
    fn m6(&self, arg: Foo<'b>) { } //~ ERROR undeclared lifetime
}

fn bar<'a>(x: &'a isize) {
    // &'a is visible to code:
    let y: &'a isize = x;

    // &'a is not visible to *items*:
    type X = Option<&'a isize>; //~ ERROR can't use generic parameters from outer item
    enum E {
        E1(&'a isize) //~ ERROR can't use generic parameters from outer item
    }
    struct S {
        f: &'a isize //~ ERROR can't use generic parameters from outer item
    }
    fn f(a: &'a isize) { } //~ ERROR can't use generic parameters from outer item

    // &'a CAN be declared on functions and used then:
    fn g<'a>(a: &'a isize) { } // OK
    fn h(a: Box<dyn for<'a> FnOnce(&'a isize)>) { } // OK
}

// Test nesting of lifetimes in fn type declarations
fn fn_types(a: &'a isize, //~ ERROR undeclared lifetime
            b: Box<dyn for<'a> FnOnce(&'a isize,
                                  &'b isize, //~ ERROR undeclared lifetime
                                  Box<dyn for<'b> FnOnce(&'a isize,
                                                     &'b isize)>,
                                  &'b isize)>, //~ ERROR undeclared lifetime
            c: &'a isize) //~ ERROR undeclared lifetime
{
}

struct Bug {}
impl Bug {
    async fn buggy(&self) -> &'a str { //~ ERROR use of undeclared lifetime name `'a`
        todo!()
    }
}

pub fn main() {}
