// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Foo;

impl Foo {
    fn bar(self) -> Foo {
        Foo
    }

    fn baz<const N: usize>(self) -> Foo {
        println!("baz: {}", N);
        Foo
    }
}

fn main() {
    Foo.bar().bar().bar().bar().baz(); //~ ERROR type annotations needed
}
