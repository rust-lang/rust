//@ revisions: old new
//@[old] edition:2015
//@[new] edition:2021
trait B { fn f(a: A) -> A; }
//~^ ERROR the trait `A` cannot be made into an object
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
trait A { fn g(b: B) -> B; }
//~^ ERROR the trait `B` cannot be made into an object
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
fn main() {}
