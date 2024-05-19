fn main() {
    let "a".. = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
    let .."a" = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
    let ..="a" = "a"; //~ ERROR only `char` and numeric types are allowed in range patterns
}
