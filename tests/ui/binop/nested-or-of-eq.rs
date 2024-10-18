fn main() {
    let x = 1;
    if x == 1 || 2 || 3 {
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        println!("Was 1 or 2 or 3");
    }

    let x = 1.0;
    if x == 1.0 || 2.0 || 3.0 {
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        println!("Was 1.0 or 2.0 or 3.0");
    }
}
