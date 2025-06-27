#![feature(const_trait_impl)]

#[const_trait]
trait Trait {}

// Regression test for issue #90052.
fn non_const_function<T: [const] Trait>() {} //~ ERROR `[const]` is not allowed

struct Struct<T: [const] Trait> { field: T } //~ ERROR `[const]` is not allowed here
struct TupleStruct<T: [const] Trait>(T); //~ ERROR `[const]` is not allowed here
struct UnitStruct<T: [const] Trait>; //~ ERROR `[const]` is not allowed here
//~^ ERROR  parameter `T` is never used

enum Enum<T: [const] Trait> { Variant(T) } //~ ERROR `[const]` is not allowed here

union Union<T: [const] Trait> { field: T } //~ ERROR `[const]` is not allowed here
//~^ ERROR field must implement `Copy`

type Type<T: [const] Trait> = T; //~ ERROR `[const]` is not allowed here

const CONSTANT<T: [const] Trait>: () = (); //~ ERROR `[const]` is not allowed here
//~^ ERROR generic const items are experimental

trait NonConstTrait {
    type Type<T: [const] Trait>: [const] Trait;
    //~^ ERROR `[const]` is not allowed
    //~| ERROR `[const]` is not allowed
    fn non_const_function<T: [const] Trait>(); //~ ERROR `[const]` is not allowed
    const CONSTANT<T: [const] Trait>: (); //~ ERROR `[const]` is not allowed
    //~^ ERROR generic const items are experimental
}

impl NonConstTrait for () {
    type Type<T: [const] Trait> = (); //~ ERROR `[const]` is not allowed
    //~^ ERROR overflow evaluating the requirement `(): Trait`
    fn non_const_function<T: [const] Trait>() {} //~ ERROR `[const]` is not allowed
    const CONSTANT<T: [const] Trait>: () = (); //~ ERROR `[const]` is not allowed
    //~^ ERROR generic const items are experimental
}

struct Implementor;

impl Implementor {
    type Type<T: [const] Trait> = (); //~ ERROR `[const]` is not allowed
    //~^ ERROR inherent associated types are unstable
    fn non_const_function<T: [const] Trait>() {} //~ ERROR `[const]` is not allowed
    const CONSTANT<T: [const] Trait>: () = (); //~ ERROR `[const]` is not allowed
    //~^ ERROR generic const items are experimental
}

// non-const traits
trait Child0: [const] Trait {} //~ ERROR `[const]` is not allowed
trait Child1 where Self: [const] Trait {} //~ ERROR `[const]` is not allowed

// non-const impl
impl<T: [const] Trait> Trait for T {} //~ ERROR `[const]` is not allowed

// inherent impl (regression test for issue #117004)
impl<T: [const] Trait> Struct<T> {} //~ ERROR `[const]` is not allowed

fn main() {}
