// compile-flags: -D path-statement
fn main() {

    let x = 10;
    x; //~ ERROR path statement with no effect
}