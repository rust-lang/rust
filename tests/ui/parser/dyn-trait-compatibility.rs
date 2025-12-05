//@ edition: 2015

type A0 = dyn;
//~^ ERROR cannot find type `dyn` in this scope
type A1 = dyn::dyn;
//~^ ERROR use of unresolved module or unlinked crate `dyn`
type A2 = dyn<dyn, dyn>;
//~^ ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
type A3 = dyn<<dyn as dyn>::dyn>;
//~^ ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find trait `dyn` in this scope

fn main() {}
