struct X(i32,i32,i32);

fn main() {
    let a = X(1, 2, 3);
    let b = a.1suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    println!("{}", b);
    let c = (1, 2, 3);
    let d = c.1suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    println!("{}", d);
    let s = X { 0suffix: 0, 1: 1, 2: 2 };
    //~^ ERROR suffixes on a tuple index are invalid
    match s {
        X { 0suffix: _, .. } => {}
        //~^ ERROR suffixes on a tuple index are invalid
    }
}
