fn main() {
    let x @ y = 0; //~ ERROR pattern bindings after an `@` are unstable
}
