fn main() {
    x = x = x;
    //~^ ERROR cannot find value `x`
    //~| ERROR cannot find value `x`
    //~| ERROR cannot find value `x`

    x = y = y = y;
    //~^ ERROR cannot find value `y`
    //~| ERROR cannot find value `y`
    //~| ERROR cannot find value `y`
    //~| ERROR cannot find value `x`

    x = y = y;
    //~^ ERROR cannot find value `x`
    //~| ERROR cannot find value `y`
    //~| ERROR cannot find value `y`

    x = x = y;
    //~^ ERROR cannot find value `x`
    //~| ERROR cannot find value `x`
    //~| ERROR cannot find value `y`

    x = x; // will suggest add `let`
    //~^ ERROR cannot find value `x`
    //~| ERROR cannot find value `x`

    x = y // will suggest add `let`
    //~^ ERROR cannot find value `x`
    //~| ERROR cannot find value `y`
}
