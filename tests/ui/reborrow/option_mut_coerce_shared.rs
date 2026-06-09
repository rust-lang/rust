fn method(a: Option<&()>) {}  //~NOTE function defined here

fn main() {
    let a = Some(&mut ());
    method(a);
    //~^ ERROR mismatched types
    //~| NOTE arguments to this function are incorrect
    //~| NOTE types differ in mutability
    //~| NOTE expected enum `Option<&()>`
    //~| NOTE    found enum `Option<&mut ()>`
}
