//@ aux-lint:hello_lint.rs

fn main() {
    let _a = 1+1; //~ ERROR this is a binary expression
}
