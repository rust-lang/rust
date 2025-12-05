fn main() {
    x = x = x;
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `x` in this scope

    x = y = y = y;
    //~^ ERROR cannot find value `y` in this scope
    //~| ERROR cannot find value `y` in this scope
    //~| ERROR cannot find value `y` in this scope
    //~| ERROR cannot find value `x` in this scope

    x = y = y;
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope
    //~| ERROR cannot find value `y` in this scope

    x = x = y;
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope

    x = x; // will suggest add `let`
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `x` in this scope

    x = y // will suggest add `let`
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope
}
