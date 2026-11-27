//@ revisions: real pre_expansion
//@[pre_expansion] check-pass
//@ edition: 2021

#[cfg(real)]
trait Foo: use<> {
    //[real]~^ ERROR `use<...>` precise capturing syntax not allowed
    type Assoc: use<> where (): use<>;
    //[real]~^ ERROR `use<...>` precise capturing syntax not allowed
    //[real]~| ERROR `use<...>` precise capturing syntax not allowed
}

#[cfg(real)]
fn fun<T: use<>>(_: impl use<>) where (): use<> {}
//[real]~^ ERROR `use<...>` precise capturing syntax not allowed
//[real]~| ERROR `use<...>` precise capturing syntax not allowed
//[real]~| ERROR `use<...>` precise capturing syntax not allowed
//[real]~| ERROR at least one trait must be specified

#[cfg(real)]
fn dynamic() -> Box<dyn use<>> {}
//[real]~^ ERROR `use<...>` precise capturing syntax not allowed in `dyn` trait object bounds
//[real]~| ERROR at least one trait is required for an object type [E0224]

fn main() {}
