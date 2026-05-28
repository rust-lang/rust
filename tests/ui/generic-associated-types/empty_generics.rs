trait Foo {
    type Bar<,>;
    //~^ ERROR expected one of `#`, `>`, `const`, identifier, or lifetime, found `,`
}

fn main() {}
