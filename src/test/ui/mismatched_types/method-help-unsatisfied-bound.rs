struct Foo;

fn main() {
    let a: Result<(), Foo> = Ok(());
    a.unwrap();
    //~^ ERROR no method named `unwrap` found
}
