fn main() {
    #[derive(Debug)]
    enum Foo {
        //~^ HELP consider annotating `Foo` with `#[derive(PartialEq)]`
        Bar,
        Qux,
    }

    let vec1 = vec![Foo::Bar, Foo::Qux];
    let vec2 = vec![Foo::Bar, Foo::Qux];
    assert_eq!(vec1, vec2);
    //~^ ERROR binary operation `==` cannot be applied to type `Vec<Foo>`
}
