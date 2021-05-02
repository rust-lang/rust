#![feature(trait_alias)]
#![feature(ptr_metadata)]

#![crate_name = "foo"]

// @has foo/fn.this_never_panics.html '//a[@title="traitalias core::ptr::metadata::Thin"]' 'Thin'
pub fn this_never_panics<T: std::ptr::Thin>() {
    assert_eq!(std::mem::size_of::<&T>(), std::mem::size_of::<usize>())
}
