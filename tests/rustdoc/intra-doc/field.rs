//@ has field/index.html '//a[@href="{{channel}}/core/ops/range/struct.Range.html#structfield.start"]' 'start'
//@ has field/index.html '//a[@href="{{channel}}/std/io/error/enum.ErrorKind.html#variant.NotFound"]' 'not_found'
//@ has field/index.html '//a[@href="struct.FieldAndMethod.html#structfield.x"]' 'x'
//@ has field/index.html '//a[@href="enum.VariantAndMethod.html#variant.X"]' 'X'
//! [start][std::ops::Range::start]
//! [not_found][std::io::ErrorKind::NotFound]
//! [x][field@crate::FieldAndMethod::x]
//! [X][variant@crate::VariantAndMethod::X]

pub struct FieldAndMethod {
    pub x: i32,
}

impl FieldAndMethod {
    pub fn x(&self) {}
}

pub enum VariantAndMethod {
    X {},
}

impl VariantAndMethod {
    fn X() {}
}
