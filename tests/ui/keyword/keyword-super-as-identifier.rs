fn main() {
    let super = 22; //~ ERROR cannot find item `super`
    //~^ NOTE can't use `super` as an identifier
    //~| HELP if you still want to call your identifier `super`, use the raw identifier format
}
