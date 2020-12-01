#![feature(half_open_range_patterns)]
#![feature(exclusive_range_pattern)]

fn main() {
    let "a".. = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
    let .."a" = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
    let ..="a" = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
}
