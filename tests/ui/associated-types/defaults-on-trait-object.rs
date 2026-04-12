#![feature(associated_type_defaults)]

trait Foo {
    type Assoc: Default = ();

    fn foo(&self) -> Self::Assoc {
        Default::default()
    }
}

// The assoc type constraint can be omitted for the assoc type with a default.
type FooObj = Box<dyn Foo>;

trait Bar {
    type Assoc1 = i32;
    type Assoc2;
}

// We can't omit assoc type constraints for an assoc type without a default value.
type BarObj1 = Box<dyn Bar>;
//~^ ERROR: the value of the associated type `Assoc2` in `Bar` must be specified

type BarObj2 = Box<dyn Bar<Assoc2 = ()>>;

type BarObj3 = Box<dyn Bar<Assoc1 = u32, Assoc2 = i32>>;

fn check_projs(foo1: &dyn Foo, foo2: &dyn Foo<Assoc = bool>) {
    let _: () = foo1.foo();
    let _: () = foo2.foo();
    //~^ ERROR: mismatched types
}

fn main() {}
