// Regression test that f64 exports things properly

import io::println;

fn main() {

    let digits: uint = 10 as uint;

    println(float::to_str(f64::sqrt(42.0f64) as float, digits));
}