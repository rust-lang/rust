// compile-flags: -D path-statements
fn main() {

    let x = 10;
    x; //~ ERROR path statement with no effect
}
