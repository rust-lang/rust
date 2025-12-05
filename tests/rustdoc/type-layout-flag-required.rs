// Tests that `--show-type-layout` is required in order to show layout info.

//@ !hasraw type_layout_flag_required/struct.Foo.html 'Size: '
pub struct Foo(usize);
