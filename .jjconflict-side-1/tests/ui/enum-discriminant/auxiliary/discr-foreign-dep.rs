#[derive(Default)]
pub enum Foo {
    A(u32),
    #[default]
    B,
    C(u32),
}
