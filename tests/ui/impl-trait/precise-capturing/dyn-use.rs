//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
fn dyn() -> &'static dyn use<> { &() }
//[edition2015]~^ ERROR expected one of `!`, `(`, `::`, `<`, `where`, or `{`, found keyword `use`
//[edition2024]~^^ ERROR expected identifier, found keyword `dyn`
//[edition2024]~| ERROR `use<...>` precise capturing syntax not allowed in `dyn` trait object bounds
//[edition2024]~| ERROR at least one trait is required for an object type

fn main() {}
