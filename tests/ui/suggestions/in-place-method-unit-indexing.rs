fn main() {
    let mut values = vec![3, 1, 2];
    let sorted = values.sort();
    println!("{}", sorted[0]);
    //~^ ERROR cannot index into a value of type `()`
}
