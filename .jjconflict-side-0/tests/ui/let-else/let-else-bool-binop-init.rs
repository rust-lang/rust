//@ run-rustfix



fn main() {
    let true = true && false else { return }; //~ ERROR a `&&` expression cannot be directly assigned in `let...else`
    let true = true || false else { return }; //~ ERROR a `||` expression cannot be directly assigned in `let...else`
}
