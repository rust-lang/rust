fn main() {
    1 = 2; //~ ERROR invalid left-hand side of assignment
    1 += 2; //~ ERROR invalid left-hand side of assignment
    (1, 2) = (3, 4);
    //~^ ERROR invalid left-hand side of assignment
    //~| ERROR invalid left-hand side of assignment
}
