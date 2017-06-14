// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The `Index` trait is used to specify the functionality of indexing operations
/// like `container[index]` when used in an immutable context.
///
/// `container[index]` is actually syntactic sugar for `*container.index(index)`,
/// but only when used as an immutable value. If a mutable value is requested,
/// [`IndexMut`] is used instead. This allows nice things such as
/// `let value = v[index]` if `value` implements [`Copy`].
///
/// [`IndexMut`]: ../../std/ops/trait.IndexMut.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
///
/// # Examples
///
/// The following example implements `Index` on a read-only `NucleotideCount`
/// container, enabling individual counts to be retrieved with index syntax.
///
/// ```
/// use std::ops::Index;
///
/// enum Nucleotide {
///     A,
///     C,
///     G,
///     T,
/// }
///
/// struct NucleotideCount {
///     a: usize,
///     c: usize,
///     g: usize,
///     t: usize,
/// }
///
/// impl Index<Nucleotide> for NucleotideCount {
///     type Output = usize;
///
///     fn index(&self, nucleotide: Nucleotide) -> &usize {
///         match nucleotide {
///             Nucleotide::A => &self.a,
///             Nucleotide::C => &self.c,
///             Nucleotide::G => &self.g,
///             Nucleotide::T => &self.t,
///         }
///     }
/// }
///
/// let nucleotide_count = NucleotideCount {a: 14, c: 9, g: 10, t: 12};
/// assert_eq!(nucleotide_count[Nucleotide::A], 14);
/// assert_eq!(nucleotide_count[Nucleotide::C], 9);
/// assert_eq!(nucleotide_count[Nucleotide::G], 10);
/// assert_eq!(nucleotide_count[Nucleotide::T], 12);
/// ```
#[lang = "index"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Index<Idx: ?Sized> {
    /// The returned type after indexing
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output: ?Sized;

    /// The method for the indexing (`container[index]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index(&self, index: Idx) -> &Self::Output;
}

/// The `IndexMut` trait is used to specify the functionality of indexing
/// operations like `container[index]` when used in a mutable context.
///
/// `container[index]` is actually syntactic sugar for
/// `*container.index_mut(index)`, but only when used as a mutable value. If
/// an immutable value is requested, the [`Index`] trait is used instead. This
/// allows nice things such as `v[index] = value` if `value` implements [`Copy`].
///
/// [`Index`]: ../../std/ops/trait.Index.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
///
/// # Examples
///
/// A very simple implementation of a `Balance` struct that has two sides, where
/// each can be indexed mutably and immutably.
///
/// ```
/// use std::ops::{Index,IndexMut};
///
/// #[derive(Debug)]
/// enum Side {
///     Left,
///     Right,
/// }
///
/// #[derive(Debug, PartialEq)]
/// enum Weight {
///     Kilogram(f32),
///     Pound(f32),
/// }
///
/// struct Balance {
///     pub left: Weight,
///     pub right:Weight,
/// }
///
/// impl Index<Side> for Balance {
///     type Output = Weight;
///
///     fn index<'a>(&'a self, index: Side) -> &'a Weight {
///         println!("Accessing {:?}-side of balance immutably", index);
///         match index {
///             Side::Left => &self.left,
///             Side::Right => &self.right,
///         }
///     }
/// }
///
/// impl IndexMut<Side> for Balance {
///     fn index_mut<'a>(&'a mut self, index: Side) -> &'a mut Weight {
///         println!("Accessing {:?}-side of balance mutably", index);
///         match index {
///             Side::Left => &mut self.left,
///             Side::Right => &mut self.right,
///         }
///     }
/// }
///
/// fn main() {
///     let mut balance = Balance {
///         right: Weight::Kilogram(2.5),
///         left: Weight::Pound(1.5),
///     };
///
///     // In this case balance[Side::Right] is sugar for
///     // *balance.index(Side::Right), since we are only reading
///     // balance[Side::Right], not writing it.
///     assert_eq!(balance[Side::Right],Weight::Kilogram(2.5));
///
///     // However in this case balance[Side::Left] is sugar for
///     // *balance.index_mut(Side::Left), since we are writing
///     // balance[Side::Left].
///     balance[Side::Left] = Weight::Kilogram(3.0);
/// }
/// ```
#[lang = "index_mut"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be mutably indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    /// The method for the mutable indexing (`container[index]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}
