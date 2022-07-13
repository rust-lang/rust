fn bug<'a>()
where
    [(); { //~ ERROR mismatched types
        let _: &'a (); //~ ERROR a non-static lifetime is not allowed in a `const`
    }]:
{}

fn main() {}
