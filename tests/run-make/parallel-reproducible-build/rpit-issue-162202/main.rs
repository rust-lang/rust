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

fn main() {}
