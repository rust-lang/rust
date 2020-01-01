fn main() {
    #[attr] if true {};
    //~^ ERROR cannot find attribute
    //~| ERROR attributes are not yet allowed on `if` expressions
    #[attr] if true {};
    //~^ ERROR cannot find attribute
    //~| ERROR attributes are not yet allowed on `if` expressions
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
