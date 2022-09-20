enum E {
    One(i32, i32)
}
fn main() {
    let var = E::One;
    if let E::One(var1, var2) = var {
    //~^ ERROR 0308
        println!("{var1} {var2}");
    }
}
