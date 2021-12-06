fn main() {
    0: u8<e<5>=e>
    //~^ ERROR: cannot find type `e` in this scope [E0412]
    //~| ERROR: associated type bindings are not allowed here [E0229]
    //~| ERROR: mismatched types [E0308]
}
