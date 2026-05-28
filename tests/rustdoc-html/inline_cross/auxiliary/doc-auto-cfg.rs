//@ compile-flags: --cfg extension

#[cfg(extension)]
pub fn compute() {}

pub struct Type;

impl Type {
    #[cfg(extension)]
    pub fn transform(self) -> Self { self }
}
