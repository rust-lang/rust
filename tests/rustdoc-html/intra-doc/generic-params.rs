// ignore-tidy-linelength

#![crate_name = "foo"]
#![allow(rustdoc::redundant_explicit_links)]

//! Here's a link to [`Vec<T>`] and one to [`Box<Vec<Option<T>>>`].
//! Here's a link to [`Iterator<Box<T>>::Item`].
//!
//@ has foo/index.html '//a[@href="{{channel}}/alloc/vec/struct.Vec.html"]' 'Vec<T>'
//@ has foo/index.html '//a[@href="{{channel}}/alloc/boxed/struct.Box.html"]' 'Box<Vec<Option<T>>>'
//@ has foo/index.html '//a[@href="{{channel}}/core/iter/traits/iterator/trait.Iterator.html#associatedtype.Item"]' 'Iterator<Box<T>>::Item'

//! And what about a link to [just `Option`](Option) and, [with the generic, `Option<T>`](Option<T>)?
//!
//@ has foo/index.html '//a[@href="{{channel}}/core/option/enum.Option.html"]' 'just Option'
//@ has foo/index.html '//a[@href="{{channel}}/core/option/enum.Option.html"]' 'with the generic, Option<T>'

//! We should also try linking to [`Result<T, E>`]; it has *two* generics!
//! And [`Result<T, !>`] and [`Result<!, E>`].
//!
//@ has foo/index.html '//a[@href="{{channel}}/core/result/enum.Result.html"]' 'Result<T, E>'
//@ has foo/index.html '//a[@href="{{channel}}/core/result/enum.Result.html"]' 'Result<T, !>'
//@ has foo/index.html '//a[@href="{{channel}}/core/result/enum.Result.html"]' 'Result<!, E>'

//! Now let's test a trickier case: [`Vec::<T>::new`], or you could write it
//! [with parentheses as `Vec::<T>::new()`][Vec::<T>::new()].
//! And what about something even harder? That would be [`Vec::<Box<T>>::new()`].
//!
//@ has foo/index.html '//a[@href="{{channel}}/alloc/vec/struct.Vec.html#method.new"]' 'Vec::<T>::new'
//@ has foo/index.html '//a[@href="{{channel}}/alloc/vec/struct.Vec.html#method.new"]' 'with parentheses as Vec::<T>::new()'
//@ has foo/index.html '//a[@href="{{channel}}/alloc/vec/struct.Vec.html#method.new"]' 'Vec::<Box<T>>::new()'

//! This is also pretty tricky: [`TypeId::of::<String>()`].
//! And this too: [`Vec::<std::error::Error>::len`].
//!
//@ has foo/index.html '//a[@href="{{channel}}/core/any/struct.TypeId.html#method.of"]' 'TypeId::of::<String>()'
//@ has foo/index.html '//a[@href="{{channel}}/alloc/vec/struct.Vec.html#method.len"]' 'Vec::<std::error::Error>::len'

//! We unofficially and implicitly support things that aren't valid in the actual Rust syntax, like
//! [`Box::<T>new()`]. We may not support them in the future!
//!
//@ has foo/index.html '//a[@href="{{channel}}/alloc/boxed/struct.Box.html#method.new"]' 'Box::<T>new()'

//! These will be resolved as regular links:
//! - [`this is <invalid syntax> first`](https://www.rust-lang.org)
//! - [`this is <invalid syntax> twice`]
//! - [`<invalid syntax> thrice`](https://www.rust-lang.org)
//! - [`<invalid syntax> four times`][rlo]
//! - [a < b][rlo]
//! - [c > d]
//!
//! [`this is <invalid syntax> twice`]: https://www.rust-lang.org
//! [rlo]: https://www.rust-lang.org
//! [c > d]: https://www.rust-lang.org
//!
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' 'this is <invalid syntax> first'
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' 'this is <invalid syntax> twice'
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' '<invalid syntax> thrice'
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' '<invalid syntax> four times'
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' 'a < b'
//@ has foo/index.html '//a[@href="https://www.rust-lang.org"]' 'c > d'

use std::any::TypeId;
