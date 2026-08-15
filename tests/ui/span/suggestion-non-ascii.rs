fn main() {
    let tup = (1,);
    println!("☃{}", tup[0]); //~ ERROR cannot index into a value of type
}
