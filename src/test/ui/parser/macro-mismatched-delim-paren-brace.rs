fn main() {
    foo! (
        bar, "baz", 1, 2.0
    } //~ ERROR incorrect close delimiter
} //~ ERROR unexpected close delimiter: `}`
