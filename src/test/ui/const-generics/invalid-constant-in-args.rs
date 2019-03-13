fn main() {
    let _: Vec<&str, "a"> = Vec::new(); //~ ERROR wrong number of const arguments
}
