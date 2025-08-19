fn main() {
    assert!("foo"); //~ ERROR mismatched types
    //~^ NOTE expected `bool`, found `str`
    //~| NOTE in this expansion of assert!
    let x = Some(&1);
    assert!(x); //~ ERROR mismatched types
    //~^ NOTE expected `bool`, found `Option<&{integer}>`
    //~| NOTE expected enum `bool`
    //~| NOTE in this expansion of assert!
    //~| NOTE in this expansion of assert!
    assert!(x, ""); //~ ERROR mismatched types
    //~^ NOTE expected `bool`, found `Option<&{integer}>`
    //~| NOTE expected enum `bool`
    //~| NOTE in this expansion of assert!
    //~| NOTE in this expansion of assert!
}
