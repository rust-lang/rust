struct Foo<
    'a,
    const N: usize = {
        let x: &'a ();
        //~^ ERROR generic parameters may not be used in const operations
        3
    },
>(&'a ());

fn main() {}
