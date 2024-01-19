trait B { fn f(a: A) -> A; }
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition
//~| ERROR the trait `A` cannot be made into an object
trait A { fn g(b: B) -> B; }
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition
//~| ERROR the trait `B` cannot be made into an object
fn main() {}
