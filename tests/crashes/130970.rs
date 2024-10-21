//@ known-bug: #130970
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir

fn main() {
    extern "C" {
        static symbol: [usize];
    }
    println!("{}", symbol[0]);
}
