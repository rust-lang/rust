struct X(i32,i32,i32);

fn main() {
    let a = X(1, 2, 3);
    let b = a.1suffix;
    //~^ ERROR tuple index with a suffix is invalid
    println!("{}", b);
}

