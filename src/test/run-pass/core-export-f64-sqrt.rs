// Regression test that f64 exports things properly

use std;
import std::io::println;

fn main() {

    let digits: uint = 10 as uint;

    println( float::to_str( f64::sqrt(42.0), digits) );
}