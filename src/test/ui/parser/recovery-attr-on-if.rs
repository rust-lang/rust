fn main() {
    #[attr] if true {};
    //~^ ERROR cannot find attribute
    #[attr] if true {};
    //~^ ERROR cannot find attribute
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
