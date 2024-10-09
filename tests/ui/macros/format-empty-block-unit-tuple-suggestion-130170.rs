//@ run-rustfix

fn main() {
    let s = "123";
    println!({}, "sss", s);
    //~^ ERROR format argument must be a string literal
    println!({});
    //~^ ERROR format argument must be a string literal
    println!(s, "sss", s, {});
    //~^ ERROR format argument must be a string literal
    println!((), s, {});
    //~^ ERROR format argument must be a string literal
}
