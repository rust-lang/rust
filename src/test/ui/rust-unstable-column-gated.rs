fn main() {
    println!("{}", __rust_unstable_column!());
    //~^ERROR the __rust_unstable_column macro is unstable
}
