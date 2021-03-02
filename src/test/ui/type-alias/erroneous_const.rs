type Foo = [u8; 0 - 1];
//~^ ERROR any use of this value will cause an error
//~| WARN will become a hard error

fn foo() -> Foo {
    todo!()
}
struct Q<const N: usize>;
type Bar = Q<{ 0 - 1 }>;
//~^ ERROR any use of this value will cause an error
//~| WARN will become a hard error

impl Default for Bar {
    fn default() -> Self {
        Q
    }
}

fn main() {}
