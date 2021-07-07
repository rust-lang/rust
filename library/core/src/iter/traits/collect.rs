/// Conversion from an [`Iterator`].
///
/// By implementing `FromIterator` for a type, you define how it will be
/// created from an iterator. This is common for types which describe a
/// collection of some kind.
///
/// [`FromIterator::from_iter()`] is rarely called explicitly, and is instead
/// used through [`Iterator::collect()`] method. See [`Iterator::collect()`]'s
/// documentation for more examples.
///
/// See also: [`IntoIterator`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter::FromIterator;
///
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
/// Implementing `FromIterator` for your type:
///
/// ```
/// use std::iter::FromIterator;
///
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
    message = "a value of type `{Self}` cannot be built from an iterator \
               over elements of type `{A}`",
    label = "value of type `{Self}` cannot be built from `std::iter::Iterator<Item={A}>`"
)]
pub trait FromIterator<A>: Sized {
    /// Creates a value from an iterator.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: crate::iter
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::iter::FromIterator;
    ///
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
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
/// let v = vec![1, 2, 3];
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
///         .map(|item| format!("{:?}", item))
///         .collect()
/// }
/// ```
#[rustc_diagnostic_item = "IntoIterator"]
#[rustc_skip_array_during_method_dispatch]
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
    /// Basic usage:
    ///
    /// ```
    /// let v = vec![1, 2, 3];
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
/// assert_eq!("MyCollection([5, 6, 7, 1, 2, 3])", format!("{:?}", c));
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
    /// Basic usage:
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
}

#[stable(feature = "extend_for_unit", since = "1.28.0")]
impl Extend<()> for () {
    fn extend<T: IntoIterator<Item = ()>>(&mut self, iter: T) {
        iter.into_iter().for_each(drop)
    }
    fn extend_one(&mut self, _item: ()) {}
}
