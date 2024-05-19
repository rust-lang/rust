fn main() {
    let x = (vec![1, 2, 3], );
    drop(x.0);
    drop(x); //~ ERROR use of partially moved value
}
