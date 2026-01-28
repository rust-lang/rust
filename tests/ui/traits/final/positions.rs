#![feature(final_associated_functions)]

// Just for exercising the syntax positions
#![feature(associated_type_defaults, extern_types, inherent_associated_types)]
#![allow(incomplete_features)]

final struct Foo {}
//~^ ERROR a struct cannot be `final`

final trait Trait {
//~^ ERROR a trait cannot be `final`

    final fn method() {}
    // OK!

    final type Foo = ();
    //~^ ERROR `final` is only allowed on associated functions in traits

    final const FOO: usize = 1;
    //~^ ERROR `final` is only allowed on associated functions in traits
}

final impl Foo {
    final fn method() {}
    //~^ ERROR `final` is only allowed on associated functions in traits

    final type Foo = ();
    //~^ ERROR `final` is only allowed on associated functions in traits

    final const FOO: usize = 1;
    //~^ ERROR `final` is only allowed on associated functions in traits
}

final impl Trait for Foo {
    final fn method() {}
    //~^ ERROR `final` is only allowed on associated functions in traits

    final type Foo = ();
    //~^ ERROR `final` is only allowed on associated functions in traits

    final const FOO: usize = 1;
    //~^ ERROR `final` is only allowed on associated functions in traits
}


final fn foo() {}
//~^ ERROR `final` is only allowed on associated functions in traits

final type FooTy = ();
//~^ ERROR `final` is only allowed on associated functions in traits

final const FOO: usize = 0;
//~^ ERROR `final` is only allowed on associated functions in traits

final unsafe extern "C" {
//~^ ERROR an extern block cannot be `final`

    final fn foo_extern();
    //~^ ERROR `final` is only allowed on associated functions in traits

    final type FooExtern;
    //~^ ERROR `final` is only allowed on associated functions in traits

    final static FOO_EXTERN: usize = 0;
    //~^ ERROR a static item cannot be `final`
    //~| ERROR incorrect `static` inside `extern` block
}

fn main() {}
