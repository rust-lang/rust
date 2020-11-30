#![deny(broken_intra_doc_links)]

// ignore-tidy-linelength

// @has primitive_non_default_impl/fn.str_methods.html
/// [`str::trim`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.trim"]' 'str::trim'
/// [`str::to_lowercase`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.to_lowercase"]' 'str::to_lowercase'
/// [`str::into_boxed_bytes`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.into_boxed_bytes"]' 'str::into_boxed_bytes'
/// [`str::replace`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.replace"]' 'str::replace'
pub fn str_methods() {}

// @has primitive_non_default_impl/fn.f32_methods.html
/// [f32::powi]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f32.html#method.powi"]' 'f32::powi'
/// [f32::sqrt]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f32.html#method.sqrt"]' 'f32::sqrt'
/// [f32::mul_add]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f32.html#method.mul_add"]' 'f32::mul_add'
pub fn f32_methods() {}

// @has primitive_non_default_impl/fn.f64_methods.html
/// [`f64::powi`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi"]' 'f64::powi'
/// [`f64::sqrt`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.sqrt"]' 'f64::sqrt'
/// [`f64::mul_add`]
// @has - '//*[@href="https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.mul_add"]' 'f64::mul_add'
pub fn f64_methods() {}
