//@ check-fail

fn main() {
    let length = 3;
    let values: [i32; length] = [0; length];
    //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    //~| ERROR attempt to use a non-constant value in a constant [E0435]
    println!("{}", values.len());
}
