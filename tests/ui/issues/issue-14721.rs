fn main() {
    let foo = "str";
    println!("{}", foo.desc); //~ no field `desc` on type `&str`
}
