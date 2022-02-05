struct Foo<
    'a,
    const N: usize = {
        let x: &'a ();
        //~^ ERROR use of non-static lifetime `'a` in const generic
        3
    },
>(&'a ());

fn main() {}
