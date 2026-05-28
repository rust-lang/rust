#![feature(more_qualified_paths)]

mod foo_bar {
    pub enum Example {
        Example1 {},
        Example2 {},
    }
}

fn main() {
    foo!(crate::foo_bar::Example, Example1);

    let i1 = foo_bar::Example::Example1 {};

    assert_eq!(i1.foo_example(), 1);

    let i2 = foo_bar::Example::Example2 {};

    assert_eq!(i2.foo_example(), 2);
}

#[macro_export]
macro_rules! foo {
    ($struct:path, $variant:ident) => {
        impl $struct {
            pub fn foo_example(&self) -> i32 {
                match self {
                    <$struct>::$variant { .. } => 1,
                    _ => 2,
                }
            }
        }
    };
}
