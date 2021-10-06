// revisions:rpass1 rpass2

trait Foo {
    #[cfg(rpass1)]
    fn method1(&self) -> u32;

    fn method2(&self) -> u32;

    #[cfg(rpass2)]
    fn method1(&self) -> u32;
}

impl Foo for u32 {
    fn method1(&self) -> u32 { 17 }
    fn method2(&self) -> u32 { 42 }
}

fn main() {
    let x: &dyn Foo = &0u32;
    assert_eq!(mod1::foo(x), 17);
}

mod mod1 {
    pub fn foo(x: &dyn super::Foo) -> u32 {
        x.method1()
    }
}
