//@ known-bug: #138534
//@compile-flags: -Zunpretty=expanded
#[repr(bool)]
pub enum TopFg {
    Bar,
}
