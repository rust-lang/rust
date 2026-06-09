pub(crate) struct Crate;

#[derive(Default)]
pub struct Universe {
    pub x: i32,
    pub(crate) y: i32,
    pub(crate) z: i32,
}

impl Universe {
    pub fn f(&self) {}
    pub(crate) fn g(&self) {}
    pub(crate) fn h(&self) {}
}
