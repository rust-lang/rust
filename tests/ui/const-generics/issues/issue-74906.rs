//@ edition:2018
//@ check-pass


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
