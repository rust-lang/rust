use super::TrustedLen;

/// Conversion from an [`Iterator`].
///
/// By implementing `FromIterator` for a type, you define how it will be
/// created from an iterator. This is common for types which describe a
/// collection of some kind.
///
/// If you want to create a collection from the contents of an iterator, the
/// [`Iterator::collect()`] method is preferred. However, when you need to
/// specify the container type, [`FromIterator::from_iter()`] can be more
/// readable than using a turbofish (e.g. `::<Vec<_>>()`). See the
/// [`Iterator::collect()`] documentation for more examples of its use.
///
/// See also: [`IntoIterator`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v = Vec::from_iter(five_fives);
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Using [`Iterator::collect()`] to implicitly use `FromIterator`:
///
/// ```
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v: Vec<i32> = five_fives.collect();
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Using [`FromIterator::from_iter()`] as a more readable alternative to
/// [`Iterator::collect()`]:
///
/// ```
/// use std::collections::VecDeque;
/// let first = (0..10).collect::<VecDeque<i32>>();
/// let second = VecDeque::from_iter(0..10);
///
/// assert_eq!(first, second);
/// ```
///
/// Implementing `FromIterator` for your type:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement FromIterator
/// impl FromIterator<i32> for MyCollection {
///     fn from_iter<I: IntoIterator<Item=i32>>(iter: I) -> Self {
///         let mut c = MyCollection::new();
///
///         for i in iter {
///             c.add(i);
///         }
///
///         c
///     }
/// }
///
/// // Now we can make a new iterator...
/// let iter = (0..5).into_iter();
///
/// // ... and make a MyCollection out of it
/// let c = MyCollection::from_iter(iter);
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
///
/// // collect works too!
///
/// let iter = (0..5).into_iter();
/// let c: MyCollection = iter.collect();
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    on(
        _Self = "&[{A}]",
        message = "a slice of type `{Self}` cannot be built since we need to store the elements somewhere",
        label = "try explicitly collecting into a `Vec<{A}>`",
    ),
    on(
        all(A = "{integer}", any(_Self = "&[{integral}]",)),
        message = "a slice of type `{Self}` cannot be built since we need to store the elements somewhere",
        label = "try explicitly collecting into a `Vec<{A}>`",
    ),
    on(
        _Self = "[{A}]",
        message = "a slice of type `{Self}` cannot be built since `{Self}` has no definite size",
        label = "try explicitly collecting into a `Vec<{A}>`",
    ),
    on(
        all(A = "{integer}", any(_Self = "[{integral}]",)),
        message = "a slice of type `{Self}` cannot be built since `{Self}` has no definite size",
        label = "try explicitly collecting into a `Vec<{A}>`",
    ),
    on(
        _Self = "[{A}; _]",
        message = "an array of type `{Self}` cannot be built directly from an iterator",
        label = "try collecting into a `Vec<{A}>`, then using `.try_into()`",
    ),
    on(
        all(A = "{integer}", any(_Self = "[{integral}; _]",)),
        message = "an array of type `{Self}` cannot be built directly from an iterator",
        label = "try collecting into a `Vec<{A}>`, then using `.try_into()`",
    ),
    message = "a value of type `{Self}` cannot be built from an iterator \
               over elements of type `{A}`",
    label = "value of type `{Self}` cannot be built from `std::iter::Iterator<Item={A}>`"
)]
#[rustc_diagnostic_item = "FromIterator"]
pub trait FromIterator<A>: Sized {
    /// Creates a value from an iterator.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: crate::iter
    ///
    /// # Examples
    ///
    /// ```
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "from_iter_fn"]
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
}

/// This implementation turns an iterator of tuples into a tuple of types which implement
/// [`Default`] and [`Extend`].
///
/// This is similar to [`Iterator::unzip`], but is also composable with other [`FromIterator`]
/// implementations:
///
/// ```rust
/// # fn main() -> Result<(), core::num::ParseIntError> {
/// let string = "1,2,123,4";
///
/// let (numbers, lengths): (Vec<_>, Vec<_>) = string
///     .split(',')
///     .map(|s| s.parse().map(|n: u32| (n, s.len())))
///     .collect::<Result<_, _>>()?;
///
/// assert_eq!(numbers, [1, 2, 123, 4]);
/// assert_eq!(lengths, [1, 1, 3, 1]);
/// # Ok(()) }
/// ```
#[stable(feature = "from_iterator_for_tuple", since = "1.79.0")]
impl<A, B, AE, BE> FromIterator<(AE, BE)> for (A, B)
where
    A: Default + Extend<AE>,
    B: Default + Extend<BE>,
{
    fn from_iter<I: IntoIterator<Item = (AE, BE)>>(iter: I) -> Self {
        let mut res = <(A, B)>::default();
        res.extend(iter);

        res
    }
}

/// Conversion into an [`Iterator`].
///
/// By implementing `IntoIterator` for a type, you define how it will be
/// converted to an iterator. This is common for types which describe a
/// collection of some kind.
///
/// One benefit of implementing `IntoIterator` is that your type will [work
/// with Rust's `for` loop syntax](crate::iter#for-loops-and-intoiterator).
///
/// See also: [`FromIterator`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let v = [1, 2, 3];
/// let mut iter = v.into_iter();
///
/// assert_eq!(Some(1), iter.next());
/// assert_eq!(Some(2), iter.next());
/// assert_eq!(Some(3), iter.next());
/// assert_eq!(None, iter.next());
/// ```
/// Implementing `IntoIterator` for your type:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement IntoIterator
/// impl IntoIterator for MyCollection {
///     type Item = i32;
///     type IntoIter = std::vec::IntoIter<Self::Item>;
///
///     fn into_iter(self) -> Self::IntoIter {
///         self.0.into_iter()
///     }
/// }
///
/// // Now we can make a new collection...
/// let mut c = MyCollection::new();
///
/// // ... add some stuff to it ...
/// c.add(0);
/// c.add(1);
/// c.add(2);
///
/// // ... and then turn it into an Iterator:
/// for (i, n) in c.into_iter().enumerate() {
///     assert_eq!(i as i32, n);
/// }
/// ```
///
/// It is common to use `IntoIterator` as a trait bound. This allows
/// the input collection type to change, so long as it is still an
/// iterator. Additional bounds can be specified by restricting on
/// `Item`:
///
/// ```rust
/// fn collect_as_strings<T>(collection: T) -> Vec<String>
/// where
///     T: IntoIterator,
///     T::Item: std::fmt::Debug,
/// {
///     collection
///         .into_iter()
///         .map(|item| format!("{item:?}"))
///         .collect()
/// }
/// ```
#[rustc_diagnostic_item = "IntoIterator"]
#[rustc_on_unimplemented(
    on(
        _Self = "core::ops::range::RangeTo<Idx>",
        label = "if you meant to iterate until a value, add a starting value",
        note = "`..end` is a `RangeTo`, which cannot be iterated on; you might have meant to have a \
              bounded `Range`: `0..end`"
    ),
    on(
        _Self = "core::ops::range::RangeToInclusive<Idx>",
        label = "if you meant to iterate until a value (including it), add a starting value",
        note = "`..=end` is a `RangeToInclusive`, which cannot be iterated on; you might have meant \
              to have a bounded `RangeInclusive`: `0..=end`"
    ),
    on(
        _Self = "[]",
        label = "`{Self}` is not an iterator; try calling `.into_iter()` or `.iter()`"
    ),
    on(_Self = "&[]", label = "`{Self}` is not an iterator; try calling `.iter()`"),
    on(
        _Self = "alloc::vec::Vec<T, A>",
        label = "`{Self}` is not an iterator; try calling `.into_iter()` or `.iter()`"
    ),
    on(
        _Self = "&str",
        label = "`{Self}` is not an iterator; try calling `.chars()` or `.bytes()`"
    ),
    on(
        _Self = "alloc::string::String",
        label = "`{Self}` is not an iterator; try calling `.chars()` or `.bytes()`"
    ),
    on(
        _Self = "{integral}",
        note = "if you want to iterate between `start` until a value `end`, use the exclusive range \
              syntax `start..end` or the inclusive range syntax `start..=end`"
    ),
    on(
        _Self = "{float}",
        note = "if you want to iterate between `start` until a value `end`, use the exclusive range \
              syntax `start..end` or the inclusive range syntax `start..=end`"
    ),
    label = "`{Self}` is not an iterator",
    message = "`{Self}` is not an iterator"
)]
#[rustc_skip_during_method_dispatch(array, boxed_slice)]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IntoIterator {
    /// The type of the elements being iterated over.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Which kind of iterator are we turning this into?
    #[stable(feature = "rust1", since = "1.0.0")]
    type IntoIter: Iterator<Item = Self::Item>;

    /// Creates an iterator from a value.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: crate::iter
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [1, 2, 3];
    /// let mut iter = v.into_iter();
    ///
    /// assert_eq!(Some(1), iter.next());
    /// assert_eq!(Some(2), iter.next());
    /// assert_eq!(Some(3), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[lang = "into_iter"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_iter(self) -> Self::IntoIter;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> IntoIterator for I {
    type Item = I::Item;
    type IntoIter = I;

    #[inline]
    fn into_iter(self) -> I {
        self
    }
}

/// Extend a collection with the contents of an iterator.
///
/// Iterators produce a series of values, and collections can also be thought
/// of as a series of values. The `Extend` trait bridges this gap, allowing you
/// to extend a collection by including the contents of that iterator. When
/// extending a collection with an already existing key, that entry is updated
/// or, in the case of collections that permit multiple entries with equal
/// keys, that entry is inserted.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // You can extend a String with some chars:
/// let mut message = String::from("The first three letters are: ");
///
/// message.extend(&['a', 'b', 'c']);
///
/// assert_eq!("abc", &message[29..32]);
/// ```
///
/// Implementing `Extend`:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // since MyCollection has a list of i32s, we implement Extend for i32
/// impl Extend<i32> for MyCollection {
///
///     // This is a bit simpler with the concrete type signature: we can call
///     // extend on anything which can be turned into an Iterator which gives
///     // us i32s. Because we need i32s to put into MyCollection.
///     fn extend<T: IntoIterator<Item=i32>>(&mut self, iter: T) {
///
///         // The implementation is very straightforward: loop through the
///         // iterator, and add() each element to ourselves.
///         for elem in iter {
///             self.add(elem);
///         }
///     }
/// }
///
/// let mut c = MyCollection::new();
///
/// c.add(5);
/// c.add(6);
/// c.add(7);
///
/// // let's extend our collection with three more numbers
/// c.extend(vec![1, 2, 3]);
///
/// // we've added these elements onto the end
/// assert_eq!("MyCollection([5, 6, 7, 1, 2, 3])", format!("{c:?}"));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Extend<A> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// As this is the only required method for this trait, the [trait-level] docs
    /// contain more details.
    ///
    /// [trait-level]: Extend
    ///
    /// # Examples
    ///
    /// ```
    /// // You can extend a String with some chars:
    /// let mut message = String::from("abc");
    ///
    /// message.extend(['d', 'e', 'f'].iter());
    ///
    /// assert_eq!("abcdef", &message);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T);

    /// Extends a collection with exactly one element.
    #[unstable(feature = "extend_one", issue = "72631")]
    fn extend_one(&mut self, item: A) {
        self.extend(Some(item));
    }

    /// Reserves capacity in a collection for the given number of additional elements.
    ///
    /// The default implementation does nothing.
    #[unstable(feature = "extend_one", issue = "72631")]
    fn extend_reserve(&mut self, additional: usize) {
        let _ = additional;
    }

    /// Extends a collection with one element, without checking there is enough capacity for it.
    ///
    /// # Safety
    ///
    /// **For callers:** This must only be called when we know the collection has enough capacity
    /// to contain the new item, for example because we previously called `extend_reserve`.
    ///
    /// **For implementors:** For a collection to unsafely rely on this method's safety precondition (that is,
    /// invoke UB if they are violated), it must implement `extend_reserve` correctly. In other words,
    /// callers may assume that if they `extend_reserve`ed enough space they can call this method.

    // This method is for internal usage only. It is only on the trait because of specialization's limitations.
    #[unstable(feature = "extend_one_unchecked", issue = "none")]
    #[doc(hidden)]
    unsafe fn extend_one_unchecked(&mut self, item: A)
    where
        Self: Sized,
    {
        self.extend_one(item);
    }
}

#[stable(feature = "extend_for_unit", since = "1.28.0")]
impl Extend<()> for () {
    fn extend<T: IntoIterator<Item = ()>>(&mut self, iter: T) {
        iter.into_iter().for_each(drop)
    }
    fn extend_one(&mut self, _item: ()) {}
}

#[stable(feature = "extend_for_tuple", since = "1.56.0")]
impl<A, B, ExtendA, ExtendB> Extend<(A, B)> for (ExtendA, ExtendB)
where
    ExtendA: Extend<A>,
    ExtendB: Extend<B>,
{
    /// Allows to `extend` a tuple of collections that also implement `Extend`.
    ///
    /// See also: [`Iterator::unzip`]
    ///
    /// # Examples
    /// ```
    /// let mut tuple = (vec![0], vec![1]);
    /// tuple.extend([(2, 3), (4, 5), (6, 7)]);
    /// assert_eq!(tuple.0, [0, 2, 4, 6]);
    /// assert_eq!(tuple.1, [1, 3, 5, 7]);
    ///
    /// // also allows for arbitrarily nested tuples as elements
    /// let mut nested_tuple = (vec![1], (vec![2], vec![3]));
    /// nested_tuple.extend([(4, (5, 6)), (7, (8, 9))]);
    ///
    /// let (a, (b, c)) = nested_tuple;
    /// assert_eq!(a, [1, 4, 7]);
    /// assert_eq!(b, [2, 5, 8]);
    /// assert_eq!(c, [3, 6, 9]);
    /// ```
    fn extend<T: IntoIterator<Item = (A, B)>>(&mut self, into_iter: T) {
        let (a, b) = self;
        let iter = into_iter.into_iter();
        SpecTupleExtend::extend(iter, a, b);
    }

    fn extend_one(&mut self, item: (A, B)) {
        self.0.extend_one(item.0);
        self.1.extend_one(item.1);
    }

    fn extend_reserve(&mut self, additional: usize) {
        self.0.extend_reserve(additional);
        self.1.extend_reserve(additional);
    }

    unsafe fn extend_one_unchecked(&mut self, item: (A, B)) {
        // SAFETY: Those are our safety preconditions, and we correctly forward `extend_reserve`.
        unsafe {
            self.0.extend_one_unchecked(item.0);
            self.1.extend_one_unchecked(item.1);
        }
    }
}

fn default_extend_tuple<A, B, ExtendA, ExtendB>(
    iter: impl Iterator<Item = (A, B)>,
    a: &mut ExtendA,
    b: &mut ExtendB,
) where
    ExtendA: Extend<A>,
    ExtendB: Extend<B>,
{
    fn extend<'a, A, B>(
        a: &'a mut impl Extend<A>,
        b: &'a mut impl Extend<B>,
    ) -> impl FnMut((), (A, B)) + 'a {
        move |(), (t, u)| {
            a.extend_one(t);
            b.extend_one(u);
        }
    }

    let (lower_bound, _) = iter.size_hint();
    if lower_bound > 0 {
        a.extend_reserve(lower_bound);
        b.extend_reserve(lower_bound);
    }

    iter.fold((), extend(a, b));
}

trait SpecTupleExtend<A, B> {
    fn extend(self, a: &mut A, b: &mut B);
}

impl<A, B, ExtendA, ExtendB, Iter> SpecTupleExtend<ExtendA, ExtendB> for Iter
where
    ExtendA: Extend<A>,
    ExtendB: Extend<B>,
    Iter: Iterator<Item = (A, B)>,
{
    default fn extend(self, a: &mut ExtendA, b: &mut ExtendB) {
        default_extend_tuple(self, a, b);
    }
}

impl<A, B, ExtendA, ExtendB, Iter> SpecTupleExtend<ExtendA, ExtendB> for Iter
where
    ExtendA: Extend<A>,
    ExtendB: Extend<B>,
    Iter: TrustedLen<Item = (A, B)>,
{
    fn extend(self, a: &mut ExtendA, b: &mut ExtendB) {
        fn extend<'a, A, B>(
            a: &'a mut impl Extend<A>,
            b: &'a mut impl Extend<B>,
        ) -> impl FnMut((), (A, B)) + 'a {
            // SAFETY: We reserve enough space for the `size_hint`, and the iterator is `TrustedLen`
            // so its `size_hint` is exact.
            move |(), (t, u)| unsafe {
                a.extend_one_unchecked(t);
                b.extend_one_unchecked(u);
            }
        }

        let (lower_bound, upper_bound) = self.size_hint();

        if upper_bound.is_none() {
            // We cannot reserve more than `usize::MAX` items, and this is likely to go out of memory anyway.
            default_extend_tuple(self, a, b);
            return;
        }

        if lower_bound > 0 {
            a.extend_reserve(lower_bound);
            b.extend_reserve(lower_bound);
        }

        self.fold((), extend(a, b));
    }
}
