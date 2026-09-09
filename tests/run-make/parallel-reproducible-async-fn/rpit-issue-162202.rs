trait Foo {
    fn test() -> impl IntoIterator<Item = ()> + Send;
}

struct A;
impl Foo for A {
    fn test() -> impl IntoIterator<Item = ()> + Send {
        []
    }
}

struct B;
impl Foo for B {
    fn test() -> impl IntoIterator<Item = ()> + Send {
        []
    }
}

async fn test1(_: &'_ u8) {}
async fn test2<'s>(_: &'s u8,) {}

fn main() {}
