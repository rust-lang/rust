fn t1(_: (), _: impl Fn(bool) -> bool) {}

fn main() {
    t1(() |c| c);
    //~^ ERROR cannot find value `c` in this scope
    //~| ERROR cannot find value `c` in this scope
    //~| ERROR this function takes 2 arguments but 1 argument was supplied
    t1((()) |c| c);
    //~^ ERROR cannot find value `c` in this scope
    //~| ERROR cannot find value `c` in this scope
    //~| ERROR this function takes 2 arguments but 1 argument was supplied
    let a = ();
    t1(() |a| c);
    //~^ ERROR no implementation for `() | ()`
    //~| ERROR cannot find value `c` in this scope
    //~| ERROR this function takes 2 arguments but 1 argument was supplied
    t1(() | a);
    //~^ ERROR no implementation for `() | ()`
    //~| ERROR this function takes 2 arguments but 1 argument was supplied
}
