// https://github.com/rust-lang/rust/issues/92940
//
// Show traits implemented on fundamental types that wrap local ones.

pub struct MyType;

// @has 'impl_box/struct.MyType.html'
// @has '-' '//*[@id="impl-Iterator"]' 'impl Iterator for Box<MyType>'

impl Iterator for Box<MyType> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
