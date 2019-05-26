// Tests that one can't run a destructor twice with the repeated vector
// literal syntax.

struct Foo {
    x: isize,

}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Goodbye!");
    }
}

fn main() {
    let a = Foo { x: 3 };
    let _ = [ a; 5 ];
    //~^ ERROR repeated expression does not implement `std::marker::Copy`
}
