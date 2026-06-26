//@ revisions: old new
//@[old] edition:2015
//@[new] edition:2021
trait B { fn f(a: A) -> A; }
//[new]~^ ERROR expected a type, found a trait
//[new]~| ERROR expected a type, found a trait
//[old]~^^^ ERROR the trait `A` is not dyn compatible
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
trait A { fn g(b: B) -> B; }
//[new]~^ ERROR expected a type, found a trait
//[new]~| ERROR expected a type, found a trait
//[old]~^^^ ERROR the trait `B` is not dyn compatible
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN trait objects without an explicit `dyn` are deprecated
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
//[old]~| WARN this is accepted in the current edition
fn main() {}
