macro_rules! empty {
    () => {};
}

macro_rules! arg {
    () => {
        N
        //~^ ERROR generic parameters may not be used in const operations
        //~| ERROR generic parameters may not be used in const operations
    };
}

struct Foo<const N: usize>;
fn foo<const N: usize>() -> Foo<{ arg!{} arg!{} }> { loop {} }
fn bar<const N: usize>() -> [(); { empty!{}; N }] { loop {} }
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
