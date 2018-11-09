fn main() {
    enum Enum { One=1 }
    let xs=[0;1 as usize];
    println!("{}", xs[Enum::One as usize]); //~ ERROR the len is 1 but the index is 1
}
