//@ aux-crate:assoc_const_equality=assoc-const-equality.rs
//@ edition:2021

#![crate_name = "user"]

//@ has user/fn.accept.html
//@ has - '//pre[@class="rust item-decl"]' 'fn accept(_: impl Trait<K = 0>)'
pub use assoc_const_equality::accept;
