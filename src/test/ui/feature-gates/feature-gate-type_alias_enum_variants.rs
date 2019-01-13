enum Foo {
    Bar(i32),
    Baz { i: i32 },
}

type Alias = Foo;

fn main() {
    let t = Alias::Bar(0);
    //~^ ERROR enum variants on type aliases are experimental
    let t = Alias::Baz { i: 0 };
    //~^ ERROR enum variants on type aliases are experimental
    match t {
        Alias::Bar(_i) => {}
        //~^ ERROR enum variants on type aliases are experimental
        Alias::Baz { i: _i } => {}
        //~^ ERROR enum variants on type aliases are experimental
    }
}
