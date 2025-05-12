// issue: rust-lang/rust#127868

fn main() {
    let a = [[[[[[[[[[[[[[[[[[[[1, {, (, [,;
} //~ ERROR mismatched closing delimiter: `}`
//~ ERROR this file contains an unclosed delimiter
