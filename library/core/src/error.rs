#![doc = include_str!("error.md")]
#![stable(feature = "error_in_core", since = "1.81.0")]

use crate::any::TypeId;
use crate::fmt::{self, Debug, Display, Formatter};

/// `Error` is a trait representing the basic expectations for error values,
/// i.e., values of type `E` in [`Result<T, E>`].
///
/// Errors must describe themselves through the [`Display`] and [`Debug`]
/// traits. Error messages are typically concise lowercase sentences without
/// trailing punctuation:
///
/// ```
/// let err = "NaN".parse::<u32>().unwrap_err();
/// assert_eq!(err.to_string(), "invalid digit found in string");
/// ```
///
/// Errors may provide cause information. [`Error::source()`] is generally
/// used when errors cross "abstraction boundaries". If one module must report
/// an error that is caused by an error from a lower-level module, it can allow
/// accessing that error via [`Error::source()`]. This makes it possible for the
/// high-level module to provide its own errors while also revealing some of the
/// implementation for debugging.
///
/// # Example
///
/// Implementing the `Error` trait only requires that `Debug` and `Display` are implemented too.
///
/// ```
/// use std::error::Error;
/// use std::fmt;
/// use std::path::PathBuf;
///
/// #[derive(Debug)]
/// struct ReadConfigError {
///     path: PathBuf
/// }
///
/// impl fmt::Display for ReadConfigError {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let path = self.path.display();
///         write!(f, "unable to read configuration at {path}")
///     }
/// }
///
/// impl Error for ReadConfigError {}
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Error"]
#[rustc_has_incoherent_inherent_impls]
#[allow(multiple_supertrait_upcastable)]
pub trait Error: Debug + Display {
    /// Returns the lower-level source of this error, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// struct SuperError {
    ///     source: SuperErrorSideKick,
    /// }
    ///
    /// impl fmt::Display for SuperError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "SuperError is here!")
    ///     }
    /// }
    ///
    /// impl Error for SuperError {
    ///     fn source(&self) -> Option<&(dyn Error + 'static)> {
    ///         Some(&self.source)
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct SuperErrorSideKick;
    ///
    /// impl fmt::Display for SuperErrorSideKick {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "SuperErrorSideKick is here!")
    ///     }
    /// }
    ///
    /// impl Error for SuperErrorSideKick {}
    ///
    /// fn get_super_error() -> Result<(), SuperError> {
    ///     Err(SuperError { source: SuperErrorSideKick })
    /// }
    ///
    /// fn main() {
    ///     match get_super_error() {
    ///         Err(e) => {
    ///             println!("Error: {e}");
    ///             println!("Caused by: {}", e.source().unwrap());
    ///         }
    ///         _ => println!("No error"),
    ///     }
    /// }
    /// ```
    #[stable(feature = "error_source", since = "1.30.0")]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }

    /// Gets the `TypeId` of `self`.
    #[doc(hidden)]
    #[unstable(
        feature = "error_type_id",
        reason = "this is memory-unsafe to override in user code",
        issue = "60784"
    )]
    fn type_id(&self, _: private::Internal) -> TypeId
    where
        Self: 'static,
    {
        TypeId::of::<Self>()
    }

    /// ```
    /// if let Err(e) = "xc".parse::<u32>() {
    ///     // Print `e` itself, no need for description().
    ///     eprintln!("Error: {e}");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.42.0", note = "use the Display impl or to_string()")]
    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.33.0",
        note = "replaced by Error::source, which can support downcasting"
    )]
    #[allow(missing_docs)]
    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }

    /// Provides type-based access to context intended for error reports.
    ///
    /// Used in conjunction with [`Request::provide_value`] and [`Request::provide_ref`] to extract
    /// references to member variables from `dyn Error` trait objects.
    ///
    /// # Example
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    /// use core::fmt;
    /// use core::error::{request_ref, Request};
    ///
    /// #[derive(Debug)]
    /// enum MyLittleTeaPot {
    ///     Empty,
    /// }
    ///
    /// #[derive(Debug)]
    /// struct MyBacktrace {
    ///     // ...
    /// }
    ///
    /// impl MyBacktrace {
    ///     fn new() -> MyBacktrace {
    ///         // ...
    ///         # MyBacktrace {}
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct Error {
    ///     backtrace: MyBacktrace,
    /// }
    ///
    /// impl fmt::Display for Error {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "Example Error")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Error {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request
    ///             .provide_ref::<MyBacktrace>(&self.backtrace);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let backtrace = MyBacktrace::new();
    ///     let error = Error { backtrace };
    ///     let dyn_error = &error as &dyn std::error::Error;
    ///     let backtrace_ref = request_ref::<MyBacktrace>(dyn_error).unwrap();
    ///
    ///     assert!(core::ptr::eq(&error.backtrace, backtrace_ref));
    ///     assert!(request_ref::<MyLittleTeaPot>(dyn_error).is_none());
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    #[allow(unused_variables)]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {}
}

mod private {
    // This is a hack to prevent `type_id` from being overridden by `Error`
    // implementations, since that can enable unsound downcasting.
    #[unstable(feature = "error_type_id", issue = "60784")]
    #[derive(Debug)]
    pub struct Internal;
}

#[unstable(feature = "never_type", issue = "35121")]
impl Error for ! {}

// Copied from `any.rs`.
impl dyn Error + 'static {
    /// Returns `true` if the inner type is the same as `T`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        // Get `TypeId` of the type this function is instantiated with.
        let t = TypeId::of::<T>();

        // Get `TypeId` of the type in the trait object (`self`).
        let concrete = self.type_id(private::Internal);

        // Compare both `TypeId`s on equality.
        t == concrete
    }

    /// Returns some reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: `is` ensures this type cast is correct
            unsafe { Some(&*(self as *const dyn Error as *const T)) }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            // SAFETY: `is` ensures this type cast is correct
            unsafe { Some(&mut *(self as *mut dyn Error as *mut T)) }
        } else {
            None
        }
    }
}

impl dyn Error + 'static + Send {
    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        <dyn Error + 'static>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        <dyn Error + 'static>::downcast_mut::<T>(self)
    }
}

impl dyn Error + 'static + Send + Sync {
    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        <dyn Error + 'static>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        <dyn Error + 'static>::downcast_mut::<T>(self)
    }
}

impl dyn Error {
    /// Returns an iterator starting with the current error and continuing with
    /// recursively calling [`Error::source`].
    ///
    /// If you want to omit the current error and only use its sources,
    /// use `skip(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(error_iter)]
    /// use std::error::Error;
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// struct A;
    ///
    /// #[derive(Debug)]
    /// struct B(Option<Box<dyn Error + 'static>>);
    ///
    /// impl fmt::Display for A {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "A")
    ///     }
    /// }
    ///
    /// impl fmt::Display for B {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "B")
    ///     }
    /// }
    ///
    /// impl Error for A {}
    ///
    /// impl Error for B {
    ///     fn source(&self) -> Option<&(dyn Error + 'static)> {
    ///         self.0.as_ref().map(|e| e.as_ref())
    ///     }
    /// }
    ///
    /// let b = B(Some(Box::new(A)));
    ///
    /// // let err : Box<Error> = b.into(); // or
    /// let err = &b as &(dyn Error);
    ///
    /// let mut iter = err.sources();
    ///
    /// assert_eq!("B".to_string(), iter.next().unwrap().to_string());
    /// assert_eq!("A".to_string(), iter.next().unwrap().to_string());
    /// assert!(iter.next().is_none());
    /// assert!(iter.next().is_none());
    /// ```
    #[unstable(feature = "error_iter", issue = "58520")]
    #[inline]
    pub fn sources(&self) -> Source<'_> {
        // You may think this method would be better in the `Error` trait, and you'd be right.
        // Unfortunately that doesn't work, not because of the dyn-incompatibility rules but
        // because we save a reference to `self` in `Source`s below as a trait object.
        // If this method was declared in `Error`, then `self` would have the type `&T` where
        // `T` is some concrete type which implements `Error`. We would need to coerce `self`
        // to have type `&dyn Error`, but that requires that `Self` has a known size
        // (i.e., `Self: Sized`). We can't put that bound on `Error` since that would forbid
        // `Error` trait objects, and we can't put that bound on the method because that means
        // the method can't be called on trait objects (we'd also need the `'static` bound,
        // but that isn't allowed because methods with bounds on `Self` other than `Sized` are
        // dyn-incompatible). Requiring an `Unsize` bound is not backwards compatible.

        Source { current: Some(self) }
    }
}

/// Requests a value of type `T` from the given `impl Error`.
///
/// # Examples
///
/// Get a string value from an error.
///
/// ```rust
/// #![feature(error_generic_member_access)]
/// use std::error::Error;
/// use core::error::request_value;
///
/// fn get_string(err: &impl Error) -> String {
///     request_value::<String>(err).unwrap()
/// }
/// ```
#[unstable(feature = "error_generic_member_access", issue = "99301")]
pub fn request_value<'a, T>(err: &'a (impl Error + ?Sized)) -> Option<T>
where
    T: 'static,
{
    request_by_type_tag::<'a, tags::Value<T>>(err)
}

/// Requests a reference of type `T` from the given `impl Error`.
///
/// # Examples
///
/// Get a string reference from an error.
///
/// ```rust
/// #![feature(error_generic_member_access)]
/// use core::error::Error;
/// use core::error::request_ref;
///
/// fn get_str(err: &impl Error) -> &str {
///     request_ref::<str>(err).unwrap()
/// }
/// ```
#[unstable(feature = "error_generic_member_access", issue = "99301")]
pub fn request_ref<'a, T>(err: &'a (impl Error + ?Sized)) -> Option<&'a T>
where
    T: 'static + ?Sized,
{
    request_by_type_tag::<'a, tags::Ref<tags::MaybeSizedValue<T>>>(err)
}

/// Request a specific value by tag from the `Error`.
fn request_by_type_tag<'a, I>(err: &'a (impl Error + ?Sized)) -> Option<I::Reified>
where
    I: tags::Type<'a>,
{
    let mut tagged = Tagged { tag_id: TypeId::of::<I>(), value: TaggedOption::<'a, I>(None) };
    err.provide(tagged.as_request());
    tagged.value.0
}

///////////////////////////////////////////////////////////////////////////////
// Request and its methods
///////////////////////////////////////////////////////////////////////////////

/// `Request` supports generic, type-driven access to data. Its use is currently restricted to the
/// standard library in cases where trait authors wish to allow trait implementors to share generic
/// information across trait boundaries. The motivating and prototypical use case is
/// `core::error::Error` which would otherwise require a method per concrete type (eg.
/// `std::backtrace::Backtrace` instance that implementors want to expose to users).
///
/// # Data flow
///
/// To describe the intended data flow for Request objects, let's consider two conceptual users
/// separated by API boundaries:
///
/// * Consumer - the consumer requests objects using a Request instance; eg a crate that offers
/// fancy `Error`/`Result` reporting to users wants to request a Backtrace from a given `dyn Error`.
///
/// * Producer - the producer provides objects when requested via Request; eg. a library with an
/// an `Error` implementation that automatically captures backtraces at the time instances are
/// created.
///
/// The consumer only needs to know where to submit their request and are expected to handle the
/// request not being fulfilled by the use of `Option<T>` in the responses offered by the producer.
///
/// * A Producer initializes the value of one of its fields of a specific type. (or is otherwise
/// prepared to generate a value requested). eg, `backtrace::Backtrace` or
/// `std::backtrace::Backtrace`
/// * A Consumer requests an object of a specific type (say `std::backtrace::Backtrace`). In the
/// case of a `dyn Error` trait object (the Producer), there are functions called `request_ref` and
/// `request_value` to simplify obtaining an `Option<T>` for a given type.
/// * The Producer, when requested, populates the given Request object which is given as a mutable
/// reference.
/// * The Consumer extracts a value or reference to the requested type from the `Request` object
/// wrapped in an `Option<T>`; in the case of `dyn Error` the aforementioned `request_ref` and `
/// request_value` methods mean that `dyn Error` users don't have to deal with the `Request` type at
/// all (but `Error` implementors do). The `None` case of the `Option` suggests only that the
/// Producer cannot currently offer an instance of the requested type, not it can't or never will.
///
/// # Examples
///
/// The best way to demonstrate this is using an example implementation of `Error`'s `provide` trait
/// method:
///
/// ```
/// #![feature(error_generic_member_access)]
/// use core::fmt;
/// use core::error::Request;
/// use core::error::request_ref;
///
/// #[derive(Debug)]
/// enum MyLittleTeaPot {
///     Empty,
/// }
///
/// #[derive(Debug)]
/// struct MyBacktrace {
///     // ...
/// }
///
/// impl MyBacktrace {
///     fn new() -> MyBacktrace {
///         // ...
///         # MyBacktrace {}
///     }
/// }
///
/// #[derive(Debug)]
/// struct Error {
///     backtrace: MyBacktrace,
/// }
///
/// impl fmt::Display for Error {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "Example Error")
///     }
/// }
///
/// impl std::error::Error for Error {
///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
///         request
///             .provide_ref::<MyBacktrace>(&self.backtrace);
///     }
/// }
///
/// fn main() {
///     let backtrace = MyBacktrace::new();
///     let error = Error { backtrace };
///     let dyn_error = &error as &dyn std::error::Error;
///     let backtrace_ref = request_ref::<MyBacktrace>(dyn_error).unwrap();
///
///     assert!(core::ptr::eq(&error.backtrace, backtrace_ref));
///     assert!(request_ref::<MyLittleTeaPot>(dyn_error).is_none());
/// }
/// ```
///
#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[repr(transparent)]
pub struct Request<'a>(Tagged<dyn Erased<'a> + 'a>);

impl<'a> Request<'a> {
    /// Provides a value or other type with only static lifetimes.
    ///
    /// # Examples
    ///
    /// Provides an `u8`.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    ///
    /// #[derive(Debug)]
    /// struct SomeConcreteType { field: u8 }
    ///
    /// impl std::fmt::Display for SomeConcreteType {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{} failed", self.field)
    ///     }
    /// }
    ///
    /// impl std::error::Error for SomeConcreteType {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request.provide_value::<u8>(self.field);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn provide_value<T>(&mut self, value: T) -> &mut Self
    where
        T: 'static,
    {
        self.provide::<tags::Value<T>>(value)
    }

    /// Provides a value or other type with only static lifetimes computed using a closure.
    ///
    /// # Examples
    ///
    /// Provides a `String` by cloning.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    ///
    /// #[derive(Debug)]
    /// struct SomeConcreteType { field: String }
    ///
    /// impl std::fmt::Display for SomeConcreteType {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{} failed", self.field)
    ///     }
    /// }
    ///
    /// impl std::error::Error for SomeConcreteType {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request.provide_value_with::<String>(|| self.field.clone());
    ///     }
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn provide_value_with<T>(&mut self, fulfil: impl FnOnce() -> T) -> &mut Self
    where
        T: 'static,
    {
        self.provide_with::<tags::Value<T>>(fulfil)
    }

    /// Provides a reference. The referee type must be bounded by `'static`,
    /// but may be unsized.
    ///
    /// # Examples
    ///
    /// Provides a reference to a field as a `&str`.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    ///
    /// #[derive(Debug)]
    /// struct SomeConcreteType { field: String }
    ///
    /// impl std::fmt::Display for SomeConcreteType {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{} failed", self.field)
    ///     }
    /// }
    ///
    /// impl std::error::Error for SomeConcreteType {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request.provide_ref::<str>(&self.field);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn provide_ref<T: ?Sized + 'static>(&mut self, value: &'a T) -> &mut Self {
        self.provide::<tags::Ref<tags::MaybeSizedValue<T>>>(value)
    }

    /// Provides a reference computed using a closure. The referee type
    /// must be bounded by `'static`, but may be unsized.
    ///
    /// # Examples
    ///
    /// Provides a reference to a field as a `&str`.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    ///
    /// #[derive(Debug)]
    /// struct SomeConcreteType { business: String, party: String }
    /// fn today_is_a_weekday() -> bool { true }
    ///
    /// impl std::fmt::Display for SomeConcreteType {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{} failed", self.business)
    ///     }
    /// }
    ///
    /// impl std::error::Error for SomeConcreteType {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request.provide_ref_with::<str>(|| {
    ///             if today_is_a_weekday() {
    ///                 &self.business
    ///             } else {
    ///                 &self.party
    ///             }
    ///         });
    ///     }
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn provide_ref_with<T: ?Sized + 'static>(
        &mut self,
        fulfil: impl FnOnce() -> &'a T,
    ) -> &mut Self {
        self.provide_with::<tags::Ref<tags::MaybeSizedValue<T>>>(fulfil)
    }

    /// Provides a value with the given `Type` tag.
    fn provide<I>(&mut self, value: I::Reified) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        if let Some(res @ TaggedOption(None)) = self.0.downcast_mut::<I>() {
            res.0 = Some(value);
        }
        self
    }

    /// Provides a value with the given `Type` tag, using a closure to prevent unnecessary work.
    fn provide_with<I>(&mut self, fulfil: impl FnOnce() -> I::Reified) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        if let Some(res @ TaggedOption(None)) = self.0.downcast_mut::<I>() {
            res.0 = Some(fulfil());
        }
        self
    }

    /// Checks if the `Request` would be satisfied if provided with a
    /// value of the specified type. If the type does not match or has
    /// already been provided, returns false.
    ///
    /// # Examples
    ///
    /// Checks if a `u8` still needs to be provided and then provides
    /// it.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    /// use core::error::request_value;
    ///
    /// #[derive(Debug)]
    /// struct Parent(Option<u8>);
    ///
    /// impl std::fmt::Display for Parent {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "a parent failed")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Parent {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         if let Some(v) = self.0 {
    ///             request.provide_value::<u8>(v);
    ///         }
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct Child {
    ///     parent: Parent,
    /// }
    ///
    /// impl Child {
    ///     // Pretend that this takes a lot of resources to evaluate.
    ///     fn an_expensive_computation(&self) -> Option<u8> {
    ///         Some(99)
    ///     }
    /// }
    ///
    /// impl std::fmt::Display for Child {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "child failed: \n  because of parent: {}", self.parent)
    ///     }
    /// }
    ///
    /// impl std::error::Error for Child {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         // In general, we don't know if this call will provide
    ///         // an `u8` value or not...
    ///         self.parent.provide(request);
    ///
    ///         // ...so we check to see if the `u8` is needed before
    ///         // we run our expensive computation.
    ///         if request.would_be_satisfied_by_value_of::<u8>() {
    ///             if let Some(v) = self.an_expensive_computation() {
    ///                 request.provide_value::<u8>(v);
    ///             }
    ///         }
    ///
    ///         // The request will be satisfied now, regardless of if
    ///         // the parent provided the value or we did.
    ///         assert!(!request.would_be_satisfied_by_value_of::<u8>());
    ///     }
    /// }
    ///
    /// let parent = Parent(Some(42));
    /// let child = Child { parent };
    /// assert_eq!(Some(42), request_value::<u8>(&child));
    ///
    /// let parent = Parent(None);
    /// let child = Child { parent };
    /// assert_eq!(Some(99), request_value::<u8>(&child));
    ///
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn would_be_satisfied_by_value_of<T>(&self) -> bool
    where
        T: 'static,
    {
        self.would_be_satisfied_by::<tags::Value<T>>()
    }

    /// Checks if the `Request` would be satisfied if provided with a
    /// reference to a value of the specified type.
    ///
    /// If the type does not match or has already been provided, returns false.
    ///
    /// # Examples
    ///
    /// Checks if a `&str` still needs to be provided and then provides
    /// it.
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    ///
    /// use core::error::Request;
    /// use core::error::request_ref;
    ///
    /// #[derive(Debug)]
    /// struct Parent(Option<String>);
    ///
    /// impl std::fmt::Display for Parent {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "a parent failed")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Parent {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         if let Some(v) = &self.0 {
    ///             request.provide_ref::<str>(v);
    ///         }
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct Child {
    ///     parent: Parent,
    ///     name: String,
    /// }
    ///
    /// impl Child {
    ///     // Pretend that this takes a lot of resources to evaluate.
    ///     fn an_expensive_computation(&self) -> Option<&str> {
    ///         Some(&self.name)
    ///     }
    /// }
    ///
    /// impl std::fmt::Display for Child {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "{} failed: \n  {}", self.name, self.parent)
    ///     }
    /// }
    ///
    /// impl std::error::Error for Child {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         // In general, we don't know if this call will provide
    ///         // a `str` reference or not...
    ///         self.parent.provide(request);
    ///
    ///         // ...so we check to see if the `&str` is needed before
    ///         // we run our expensive computation.
    ///         if request.would_be_satisfied_by_ref_of::<str>() {
    ///             if let Some(v) = self.an_expensive_computation() {
    ///                 request.provide_ref::<str>(v);
    ///             }
    ///         }
    ///
    ///         // The request will be satisfied now, regardless of if
    ///         // the parent provided the reference or we did.
    ///         assert!(!request.would_be_satisfied_by_ref_of::<str>());
    ///     }
    /// }
    ///
    /// let parent = Parent(Some("parent".into()));
    /// let child = Child { parent, name: "child".into() };
    /// assert_eq!(Some("parent"), request_ref::<str>(&child));
    ///
    /// let parent = Parent(None);
    /// let child = Child { parent, name: "child".into() };
    /// assert_eq!(Some("child"), request_ref::<str>(&child));
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn would_be_satisfied_by_ref_of<T>(&self) -> bool
    where
        T: ?Sized + 'static,
    {
        self.would_be_satisfied_by::<tags::Ref<tags::MaybeSizedValue<T>>>()
    }

    fn would_be_satisfied_by<I>(&self) -> bool
    where
        I: tags::Type<'a>,
    {
        matches!(self.0.downcast::<I>(), Some(TaggedOption(None)))
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a> Debug for Request<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request").finish_non_exhaustive()
    }
}

///////////////////////////////////////////////////////////////////////////////
// Type tags
///////////////////////////////////////////////////////////////////////////////

pub(crate) mod tags {
    //! Type tags are used to identify a type using a separate value. This module includes type tags
    //! for some very common types.
    //!
    //! Currently type tags are not exposed to the user. But in the future, if you want to use the
    //! Request API with more complex types (typically those including lifetime parameters), you
    //! will need to write your own tags.

    use crate::marker::PhantomData;

    /// This trait is implemented by specific tag types in order to allow
    /// describing a type which can be requested for a given lifetime `'a`.
    ///
    /// A few example implementations for type-driven tags can be found in this
    /// module, although crates may also implement their own tags for more
    /// complex types with internal lifetimes.
    pub(crate) trait Type<'a>: Sized + 'static {
        /// The type of values which may be tagged by this tag for the given
        /// lifetime.
        type Reified: 'a;
    }

    /// Similar to the [`Type`] trait, but represents a type which may be unsized (i.e., has a
    /// `?Sized` bound). E.g., `str`.
    pub(crate) trait MaybeSizedType<'a>: Sized + 'static {
        type Reified: 'a + ?Sized;
    }

    impl<'a, T: Type<'a>> MaybeSizedType<'a> for T {
        type Reified = T::Reified;
    }

    /// Type-based tag for types bounded by `'static`, i.e., with no borrowed elements.
    #[derive(Debug)]
    pub(crate) struct Value<T: 'static>(PhantomData<T>);

    impl<'a, T: 'static> Type<'a> for Value<T> {
        type Reified = T;
    }

    /// Type-based tag similar to [`Value`] but which may be unsized (i.e., has a `?Sized` bound).
    #[derive(Debug)]
    pub(crate) struct MaybeSizedValue<T: ?Sized + 'static>(PhantomData<T>);

    impl<'a, T: ?Sized + 'static> MaybeSizedType<'a> for MaybeSizedValue<T> {
        type Reified = T;
    }

    /// Type-based tag for reference types (`&'a T`, where T is represented by
    /// `<I as MaybeSizedType<'a>>::Reified`.
    #[derive(Debug)]
    pub(crate) struct Ref<I>(PhantomData<I>);

    impl<'a, I: MaybeSizedType<'a>> Type<'a> for Ref<I> {
        type Reified = &'a I::Reified;
    }
}

/// An `Option` with a type tag `I`.
///
/// Since this struct implements `Erased`, the type can be erased to make a dynamically typed
/// option. The type can be checked dynamically using `Tagged::tag_id` and since this is statically
/// checked for the concrete type, there is some degree of type safety.
#[repr(transparent)]
pub(crate) struct TaggedOption<'a, I: tags::Type<'a>>(pub Option<I::Reified>);

impl<'a, I: tags::Type<'a>> Tagged<TaggedOption<'a, I>> {
    pub(crate) fn as_request(&mut self) -> &mut Request<'a> {
        let erased = self as &mut Tagged<dyn Erased<'a> + 'a>;
        // SAFETY: transmuting `&mut Tagged<dyn Erased<'a> + 'a>` to `&mut Request<'a>` is safe since
        // `Request` is repr(transparent).
        unsafe { &mut *(erased as *mut Tagged<dyn Erased<'a>> as *mut Request<'a>) }
    }
}

/// Represents a type-erased but identifiable object.
///
/// This trait is exclusively implemented by the `TaggedOption` type.
unsafe trait Erased<'a>: 'a {}

unsafe impl<'a, I: tags::Type<'a>> Erased<'a> for TaggedOption<'a, I> {}

struct Tagged<E: ?Sized> {
    tag_id: TypeId,
    value: E,
}

impl<'a> Tagged<dyn Erased<'a> + 'a> {
    /// Returns some reference to the dynamic value if it is tagged with `I`,
    /// or `None` otherwise.
    #[inline]
    fn downcast<I>(&self) -> Option<&TaggedOption<'a, I>>
    where
        I: tags::Type<'a>,
    {
        if self.tag_id == TypeId::of::<I>() {
            // SAFETY: Just checked whether we're pointing to an I.
            Some(&unsafe { &*(self as *const Self).cast::<Tagged<TaggedOption<'a, I>>>() }.value)
        } else {
            None
        }
    }

    /// Returns some mutable reference to the dynamic value if it is tagged with `I`,
    /// or `None` otherwise.
    #[inline]
    fn downcast_mut<I>(&mut self) -> Option<&mut TaggedOption<'a, I>>
    where
        I: tags::Type<'a>,
    {
        if self.tag_id == TypeId::of::<I>() {
            Some(
                // SAFETY: Just checked whether we're pointing to an I.
                &mut unsafe { &mut *(self as *mut Self).cast::<Tagged<TaggedOption<'a, I>>>() }
                    .value,
            )
        } else {
            None
        }
    }
}

/// An iterator over an [`Error`] and its sources.
///
/// If you want to omit the initial error and only process
/// its sources, use `skip(1)`.
#[unstable(feature = "error_iter", issue = "58520")]
#[derive(Clone, Debug)]
pub struct Source<'a> {
    current: Option<&'a (dyn Error + 'static)>,
}

#[unstable(feature = "error_iter", issue = "58520")]
impl<'a> Iterator for Source<'a> {
    type Item = &'a (dyn Error + 'static);

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        self.current = self.current.and_then(Error::source);
        current
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.current.is_some() { (1, None) } else { (0, Some(0)) }
    }
}

#[unstable(feature = "error_iter", issue = "58520")]
impl<'a> crate::iter::FusedIterator for Source<'a> {}

#[stable(feature = "error_by_ref", since = "1.51.0")]
impl<'a, T: Error + ?Sized> Error for &'a T {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        Error::description(&**self)
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Error::source(&**self)
    }

    fn provide<'b>(&'b self, request: &mut Request<'b>) {
        Error::provide(&**self, request);
    }
}

#[stable(feature = "fmt_error", since = "1.11.0")]
impl Error for crate::fmt::Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "an error occurred when formatting an argument"
    }
}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for crate::cell::BorrowError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "already mutably borrowed"
    }
}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for crate::cell::BorrowMutError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "already borrowed"
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for crate::char::CharTryFromError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "converted integer out of range for `char`"
    }
}

#[stable(feature = "duration_checked_float", since = "1.66.0")]
impl Error for crate::time::TryFromFloatSecsError {}

#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
impl Error for crate::ffi::FromBytesUntilNulError {}

#[stable(feature = "get_many_mut", since = "1.86.0")]
impl Error for crate::slice::GetDisjointMutError {}
