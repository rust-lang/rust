#![deny(non_ascii_idents)]

const חלודה: usize = 2; //~ ERROR identifier contains non-ASCII characters

fn coöperation() {} //~ ERROR identifier contains non-ASCII characters

fn main() {
    let naïveté = 2; //~ ERROR identifier contains non-ASCII characters

    // using the same identifier the second time won't trigger the lint.
    println!("{}", naïveté);
}
