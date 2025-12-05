#![crate_name = "foo"]

//@ has 'foo/index.html' '//meta[@name="description"]/@content' \
//   'API documentation for the Rust `foo` crate.'

//@ has 'foo/foo_mod/index.html' '//meta[@name="description"]/@content' \
//   'API documentation for the Rust `foo_mod` mod in crate `foo`.'
pub mod foo_mod {
    pub struct __Thing {}
}

//@ has 'foo/fn.foo_fn.html' '//meta[@name="description"]/@content' \
//   'API documentation for the Rust `foo_fn` fn in crate `foo`.'
pub fn foo_fn() {}
