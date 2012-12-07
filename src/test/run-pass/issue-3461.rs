// xfail-test
fn main() {

    fn foo() { }
    
    let bar: ~fn() = ~foo;
}