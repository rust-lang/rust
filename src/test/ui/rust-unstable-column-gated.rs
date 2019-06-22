fn main() {
    println!("{}", __rust_unstable_column!());
    //~^ ERROR use of unstable library feature '__rust_unstable_column'
}
