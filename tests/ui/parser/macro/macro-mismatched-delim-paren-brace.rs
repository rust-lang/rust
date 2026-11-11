fn main() {
    foo! (
        bar, "baz", 1, 2.0
    } //~ ERROR mismatched closing delimiter
} //~ ERROR unexpected closing delimiter: `}`
