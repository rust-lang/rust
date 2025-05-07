//@ known-bug: #130970
//@ compile-flags: -Zmir-enable-passes=+GVN -Zvalidate-mir

fn main() {
    extern "C" {
        static symbol: [usize];
    }
    println!("{}", symbol[0]);
}
