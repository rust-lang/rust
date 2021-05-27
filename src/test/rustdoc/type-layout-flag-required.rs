// Tests that `--show-type-layout` is required in order to show layout info.

// @!has type_layout_flag_required/struct.Foo.html 'Size: '
pub struct Foo(usize);
