// Test for #142064, internal error: entered unreachable code

#![crate_type = "rlib"]
trait A { fn foo() -> A; }
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| ERROR the trait `A` is not dyn compatible
trait B { fn foo() -> A; }
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| ERROR the trait `A` is not dyn compatible
