#![deny(broken_intra_doc_links)]

// ignore-tidy-linelength

// @has intra_link_primitive_non_default_impl/fn.f.html
/// [`str::trim`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.trim"]' 'str::trim'
/// [`str::to_lowercase`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.to_lowercase"]' 'str::to_lowercase'
/// [`str::into_boxed_bytes`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.into_boxed_bytes"]' 'str::into_boxed_bytes'
/// [`str::replace`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.replace"]' 'str::replace'
pub fn f() {}
