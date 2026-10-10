//@ has deprecated/type.A.html '//a[@href="{{channel}}/core/ops/range/type.Range.html#structfield.start"]' 'start'
//@ has deprecated/type.B1.html '//a[@href="{{channel}}/core/io/error/type.ErrorKind.html#variant.NotFound"]' 'not_found'
//@ has deprecated/type.B2.html '//a[@href="{{channel}}/core/io/error/type.ErrorKind.html#variant.NotFound"]' 'not_found'

#[deprecated = "[start][std::ops::Range::start]"]
pub struct A;

#[deprecated(since = "0.0.0", note = "[not_found][std::io::ErrorKind::NotFound]")]
pub struct B1;

#[deprecated(note = "[not_found][std::io::ErrorKind::NotFound]", since = "0.0.0")]
pub struct B2;
