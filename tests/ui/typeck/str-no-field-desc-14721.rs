fn main() {
    let foo = "str";
    println!("{}", foo.desc); //~ ERROR no field `desc` on type `&str`
}
