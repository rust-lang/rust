//@ build-fail

fn main() {
    enum Enum { One=1 }
    let xs=[0;1 as usize];
    println!("{}", xs[Enum::One as usize]); //~ ERROR this operation will panic at runtime
}
