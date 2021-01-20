fn main() {
    extern {
        static symbol: [usize]; //~ ERROR: the size for values of type
    }
    println!("{}", symbol[0]);
}
