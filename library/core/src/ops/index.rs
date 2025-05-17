/// Used for indexing operations (`container[index]`) in immutable contexts.
///
/// `container[index]` is actually syntactic sugar for `*container.index(index)`,
/// but only when used as an immutable value. If a mutable value is requested,
/// [`IndexMut`] is used instead. This allows nice things such as
/// `let value = v[index]` if the type of `value` implements [`Copy`].
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
///     fn index(&self, nucleotide: Nucleotide) -> &Self::Output {
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
#[diagnostic::on_unimplemented(
    message = "the type `{Self}` cannot be indexed by `{Idx}`",
    label = "`{Self}` cannot be indexed by `{Idx}`"
)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = "]")]
#[doc(alias = "[")]
#[doc(alias = "[]")]
pub trait Index<Idx: ?Sized> {
    /// The returned type after indexing.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "IndexOutput"]
    type Output: ?Sized;

    /// Performs the indexing (`container[index]`) operation.
    ///
    /// # Panics
    ///
    /// May panic if the index is out of bounds.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_no_implicit_autorefs]
    #[track_caller]
    fn index(&self, index: Idx) -> &Self::Output;
}

/// Used for indexing operations (`container[index]`) in mutable contexts.
///
/// `container[index]` is actually syntactic sugar for
/// `*container.index_mut(index)`, but only when used as a mutable value. If
/// an immutable value is requested, the [`Index`] trait is used instead. This
/// allows nice things such as `v[index] = value`.
///
/// # Examples
///
/// A very simple implementation of a `Balance` struct that has two sides, where
/// each can be indexed mutably and immutably.
///
/// ```
/// use std::ops::{Index, IndexMut};
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
///     pub right: Weight,
/// }
///
/// impl Index<Side> for Balance {
///     type Output = Weight;
///
///     fn index(&self, index: Side) -> &Self::Output {
///         println!("Accessing {index:?}-side of balance immutably");
///         match index {
///             Side::Left => &self.left,
///             Side::Right => &self.right,
///         }
///     }
/// }
///
/// impl IndexMut<Side> for Balance {
///     fn index_mut(&mut self, index: Side) -> &mut Self::Output {
///         println!("Accessing {index:?}-side of balance mutably");
///         match index {
///             Side::Left => &mut self.left,
///             Side::Right => &mut self.right,
///         }
///     }
/// }
///
/// let mut balance = Balance {
///     right: Weight::Kilogram(2.5),
///     left: Weight::Pound(1.5),
/// };
///
/// // In this case, `balance[Side::Right]` is sugar for
/// // `*balance.index(Side::Right)`, since we are only *reading*
/// // `balance[Side::Right]`, not writing it.
/// assert_eq!(balance[Side::Right], Weight::Kilogram(2.5));
///
/// // However, in this case `balance[Side::Left]` is sugar for
/// // `*balance.index_mut(Side::Left)`, since we are writing
/// // `balance[Side::Left]`.
/// balance[Side::Left] = Weight::Kilogram(3.0);
/// ```
#[lang = "index_mut"]
#[rustc_on_unimplemented(
    on(
        Self = "&str",
        note = "you can use `.chars().nth()` or `.bytes().nth()`
see chapter in The Book <https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings>"
    ),
    on(
        Self = "str",
        note = "you can use `.chars().nth()` or `.bytes().nth()`
see chapter in The Book <https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings>"
    ),
    on(
        Self = "alloc::string::String",
        note = "you can use `.chars().nth()` or `.bytes().nth()`
see chapter in The Book <https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings>"
    ),
    message = "the type `{Self}` cannot be mutably indexed by `{Idx}`",
    label = "`{Self}` cannot be mutably indexed by `{Idx}`"
)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = "[")]
#[doc(alias = "]")]
#[doc(alias = "[]")]
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    /// Performs the mutable indexing (`container[index]`) operation.
    ///
    /// # Panics
    ///
    /// May panic if the index is out of bounds.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_no_implicit_autorefs]
    #[track_caller]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}
