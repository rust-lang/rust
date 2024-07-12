type A0 = dyn;
//~^ ERROR cannot find type `dyn`
type A1 = dyn::dyn;
//~^ ERROR cannot find item `dyn`
type A2 = dyn<dyn, dyn>;
//~^ ERROR cannot find type `dyn`
//~| ERROR cannot find type `dyn`
//~| ERROR cannot find type `dyn`
type A3 = dyn<<dyn as dyn>::dyn>;
//~^ ERROR cannot find type `dyn`
//~| ERROR cannot find type `dyn`
//~| ERROR cannot find trait `dyn`

fn main() {}
