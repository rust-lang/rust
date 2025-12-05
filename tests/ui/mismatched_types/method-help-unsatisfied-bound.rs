struct Foo;

fn main() {
    let a: Result<(), Foo> = Ok(());
    a.unwrap();
    //~^ ERROR `Foo` doesn't implement `Debug`
}
