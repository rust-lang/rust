fn main() {
    extern "C" {
        static symbol: [usize]; //~ ERROR: the size for values of type
    }
    println!("{}", symbol[0]);
}
