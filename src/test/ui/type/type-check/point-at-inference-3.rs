fn main() {
    let v = Vec::new();
    v.push(0);
    //~^ NOTE this is of type `{integer}`, which makes `v` to be inferred as `Vec<{integer}>`
    v.push(0);
    v.push(""); //~ ERROR mismatched types
    //~^ NOTE expected integer, found `&str`
    //~| NOTE arguments to this function are incorrect
    //~| NOTE associated function defined here
}
