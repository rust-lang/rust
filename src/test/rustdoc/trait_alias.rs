#![feature(trait_alias)]

#![crate_name = "foo"]

use std::fmt::Debug;

// @has foo/all.html '//a[@href="traitalias.CopyAlias.html"]' 'CopyAlias'
// @has foo/all.html '//a[@href="traitalias.Alias2.html"]' 'Alias2'
// @has foo/all.html '//a[@href="traitalias.Foo.html"]' 'Foo'

// @has foo/index.html '//h2[@id="trait-aliases"]' 'Trait aliases'
// @has foo/index.html '//a[@class="traitalias"]' 'CopyAlias'
// @has foo/index.html '//a[@class="traitalias"]' 'Alias2'
// @has foo/index.html '//a[@class="traitalias"]' 'Foo'

// @has foo/traitalias.CopyAlias.html '//section[@id="main"]/pre' 'trait CopyAlias = Copy;'
pub trait CopyAlias = Copy;
// @has foo/traitalias.Alias2.html '//section[@id="main"]/pre' 'trait Alias2 = Copy + Debug;'
pub trait Alias2 = Copy + Debug;
// @has foo/traitalias.Foo.html '//section[@id="main"]/pre' 'trait Foo<T> = Into<T> + Debug;'
pub trait Foo<T> = Into<T> + Debug;
