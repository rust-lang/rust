fn main() {
    b"abc".iter().for_each(|x| x); //~ ERROR: mismatched types

    b"abc".iter().for_each(|x| dbg!(x)); //~ ERROR: mismatched types

    b"abc".iter().for_each(|x| {
        println!("{}", x);
        x //~ ERROR: mismatched types
    })
}
