struct X(i32,i32,i32);

fn main() {
    let a = X(1, 2, 3);
    let b = a.1suffix;
    //~^ ERROR suffixes on tuple indexes are invalid
    println!("{}", b);
    let c = (1, 2, 3);
    let d = c.1suffix;
    //~^ ERROR suffixes on tuple indexes are invalid
    println!("{}", d);
}

