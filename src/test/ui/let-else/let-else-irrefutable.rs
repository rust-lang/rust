// check-pass



fn main() {
    let x = 1 else { return }; //~ WARN irrefutable `let...else` pattern
}
