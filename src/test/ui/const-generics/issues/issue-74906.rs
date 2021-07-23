// edition:2018
// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

const SIZE: usize = 16;

struct Bar<const H: usize> {}

struct Foo<const H: usize> {}

impl<const H: usize> Foo<H> {
    async fn biz(_: &[[u8; SIZE]]) -> Vec<()> {
        vec![]
    }

    pub async fn baz(&self) -> Bar<H> {
        Self::biz(&vec![]).await;
        Bar {}
    }
}

fn main() { }
