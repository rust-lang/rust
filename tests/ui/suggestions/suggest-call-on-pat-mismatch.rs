enum E {
    One(i32, i32),
}

fn main() {
    let var = E::One;
    if let E::One(var1, var2) = var {
        //~^ ERROR mismatched types
        //~| HELP use parentheses to construct this tuple variant
        println!("{var1} {var2}");
    }

    let Some(x) = Some;
    //~^ ERROR mismatched types
    //~| HELP use parentheses to construct this tuple variant
}
