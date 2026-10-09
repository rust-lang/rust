// Issue #105753.
trait T {}

fn foo() -> dyn T {
    //~^ ERROR: return type cannot be a trait object without pointer indirection
    todo!()
}

fn main() {
    let x = foo();
    //~^ ERROR: the size for values of type `dyn T` cannot be known at compilation time
}
