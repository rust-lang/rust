//! Implementation of [Request]

use crate::any::TypeId;
use crate::error::Error;
use crate::fmt::{self, Debug, Formatter};
use crate::marker::PhantomData;
use crate::ptr::NonNull;

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
    let mut tagged = <Tagged<TaggedOption<'a, I>>>::new_concrete();
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
///   fancy `Error`/`Result` reporting to users wants to request a Backtrace from a given `dyn Error`.
///
/// * Producer - the producer provides objects when requested via Request; eg. a library with an
///   an `Error` implementation that automatically captures backtraces at the time instances are
///   created.
///
/// The consumer only needs to know where to submit their request and are expected to handle the
/// request not being fulfilled by the use of `Option<T>` in the responses offered by the producer.
///
/// * A Producer initializes the value of one of its fields of a specific type. (or is otherwise
///   prepared to generate a value requested). eg, `backtrace::Backtrace` or
///   `std::backtrace::Backtrace`
/// * A Consumer requests an object of a specific type (say `std::backtrace::Backtrace`). In the
///   case of a `dyn Error` trait object (the Producer), there are functions called `request_ref` and
///   `request_value` to simplify obtaining an `Option<T>` for a given type.
/// * The Producer, when requested, populates the given Request object which is given as a mutable
///   reference.
/// * The Consumer extracts a value or reference to the requested type from the `Request` object
///   wrapped in an `Option<T>`; in the case of `dyn Error` the aforementioned `request_ref` and `
///   request_value` methods mean that `dyn Error` users don't have to deal with the `Request` type at
///   all (but `Error` implementors do). The `None` case of the `Option` suggests only that the
///   Producer cannot currently offer an instance of the requested type, not it can't or never will.
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
        self.0.provide::<I>(value);
        self
    }

    /// Provides a value with the given `Type` tag, using a closure to prevent unnecessary work.
    fn provide_with<I>(&mut self, fulfil: impl FnOnce() -> I::Reified) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        self.0.provide_with::<I>(fulfil);
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
        self.0.would_be_satisfied_by::<I>()
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a> Debug for Request<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request").finish_non_exhaustive()
    }
}

/// Base case for [IntoMultiRequest].
#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[derive(Copy, Clone, Debug)]
pub struct EmptyMultiRequestBuilder;

/// Case of [IntoMultiRequest] that retrieves a type by value.
///
/// Create via [MultiRequestBuilder::with_value].
#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[derive(Copy, Clone, Debug)]
pub struct ChainValMultiRequestBuilder<T, NEXT>(PhantomData<(T, NEXT)>);

#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[derive(Copy, Clone, Debug)]
/// Case of [IntoMultiRequest] that retrieves a type by value.
///
/// Create via [MultiRequestBuilder::with_ref].
pub struct ChainRefMultiRequestBuilder<T: ?Sized, NEXT>(PhantomData<(*const T, NEXT)>);

/// Internal trait for types that represent a request for multiple provided
/// traits in parallel.
///
/// There is no need to use this trait directly, use [MultiRequestBuilder] instead.
#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[allow(private_bounds)]
pub trait IntoMultiRequest: private::IntoMultiRequestInner + 'static {}

mod private {
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    #[allow(private_bounds)]
    pub trait IntoMultiRequestInner {
        #[unstable(feature = "error_generic_member_access", issue = "99301")]
        type Request<'a>: super::Erased<'a> + MultiResponseInner<'a>
        where
            Self: 'a;
        #[unstable(feature = "error_generic_member_access", issue = "99301")]
        fn get_request<'a>() -> Self::Request<'a>;
    }

    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    #[allow(private_bounds)]
    pub trait MultiResponseInner<'a> {
        fn consume_with<I>(&mut self, fulfil: impl FnOnce(I::Reified)) -> &mut Self
        where
            I: super::tags::Type<'a>;
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl IntoMultiRequest for EmptyMultiRequestBuilder {}
#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl private::IntoMultiRequestInner for EmptyMultiRequestBuilder {
    type Request<'a> = EmptyMultiResponse;

    fn get_request<'a>() -> Self::Request<'a> {
        EmptyMultiResponse
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<T, NEXT> IntoMultiRequest for ChainValMultiRequestBuilder<T, NEXT>
where
    T: 'static,
    NEXT: IntoMultiRequest,
{
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<T, NEXT> private::IntoMultiRequestInner for ChainValMultiRequestBuilder<T, NEXT>
where
    T: 'static,
    NEXT: IntoMultiRequest,
{
    type Request<'a> = ChainValMultiResponse<'a, T, NEXT::Request<'a>>;

    fn get_request<'a>() -> Self::Request<'a> {
        ChainValMultiResponse {
            inner: ChainMultiResponse { cur: None, next: NEXT::get_request(), marker: PhantomData },
        }
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<T, NEXT> IntoMultiRequest for ChainRefMultiRequestBuilder<T, NEXT>
where
    T: ?Sized + 'static,
    NEXT: IntoMultiRequest,
{
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<T, NEXT> private::IntoMultiRequestInner for ChainRefMultiRequestBuilder<T, NEXT>
where
    T: ?Sized + 'static,
    NEXT: IntoMultiRequest,
{
    type Request<'a> = ChainRefMultiResponse<'a, T, NEXT::Request<'a>>;

    fn get_request<'a>() -> Self::Request<'a> {
        ChainRefMultiResponse {
            inner: ChainMultiResponse { cur: None, next: NEXT::get_request(), marker: PhantomData },
        }
    }
}

/// A response from an empty [MultiRequestBuilder::request]
#[unstable(
    feature = "error_generic_member_access_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[derive(Debug)]
pub struct EmptyMultiResponse;

#[derive(Debug)]
struct ChainMultiResponse<'a, I, NEXT>
where
    I: tags::Type<'a>,
{
    cur: Option<I::Reified>,
    next: NEXT,
    // Lifetime is invariant because it is used in an associated type
    marker: PhantomData<*mut &'a ()>,
}

/// A response from a [MultiRequestBuilder::request] after calling [MultiRequestBuilder::with_value].
#[unstable(
    feature = "error_generic_member_access_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[derive(Debug)]
pub struct ChainValMultiResponse<'a, T, NEXT>
where
    T: 'static,
{
    inner: ChainMultiResponse<'a, tags::Value<T>, NEXT>,
}

/// A response from a [MultiRequestBuilder::request] after calling [MultiRequestBuilder::with_ref].
#[unstable(
    feature = "error_generic_member_access_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[derive(Debug)]
pub struct ChainRefMultiResponse<'a, T, NEXT>
where
    T: 'static + ?Sized,
{
    inner: ChainMultiResponse<'a, tags::Ref<tags::MaybeSizedValue<T>>, NEXT>,
}

/// A response from a [MultiRequestBuilder]. The types returned from
/// [MultiRequestBuilder::request] implement this trait.
#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[allow(private_bounds)]
pub trait MultiResponse<'a> {
    /// Retrieve a reference with the type `R` from this multi response,
    ///
    /// The reference will be passed to `fulfil` if present. This function
    /// consumes the reference, so the next call to `retrieve_ref`
    /// with the same type will not call `fulfil`.
    ///
    /// This function returns `self` to allow easy chained use.
    ///
    /// # Examples
    ///
    /// When requesting only a single type, it is better to use
    /// [request_ref] - this is only an example.
    ///
    /// ```
    /// #![feature(error_generic_member_access)]
    /// use core::error::{Error, MultiRequestBuilder, MultiResponse};
    ///
    /// fn get_str(e: &dyn Error) -> Option<&str> {
    ///     let mut result = None;
    ///     MultiRequestBuilder::new()
    ///         .with_ref::<str>()
    ///         .request(e)
    ///         .retrieve_ref(|res| result = Some(res));
    ///     result
    /// }
    /// ```
    fn retrieve_ref<R>(&mut self, fulfil: impl FnOnce(&'a R)) -> &mut Self
    where
        R: ?Sized + 'static;

    /// Retrieve a value with the type `V` from this multi response,
    ///
    /// The value will be passed to `fulfil` if present. This function
    /// consumes the value, so the next call to `retrieve_value`
    /// with the same type will not call `fulfil`.
    ///
    /// This function returns `self` to allow easy chained use.
    ///
    /// # Examples
    ///
    /// When requesting only a single type, it is better to use
    /// [request_value] - this is only an example.
    ///
    /// ```
    /// #![feature(error_generic_member_access)]
    /// use core::error::{Error, MultiRequestBuilder, MultiResponse};
    ///
    /// fn get_string(e: &dyn Error) -> Option<String> {
    ///     let mut result = None;
    ///     MultiRequestBuilder::new()
    ///         .with_value::<String>()
    ///         .request(e)
    ///         .retrieve_value(|res| result = Some(res));
    ///     result
    /// }
    /// ```
    fn retrieve_value<V>(&mut self, fulfil: impl FnOnce(V)) -> &mut Self
    where
        V: 'static;
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a, T: private::MultiResponseInner<'a>> MultiResponse<'a> for T {
    fn retrieve_ref<R>(&mut self, fulfil: impl FnOnce(&'a R)) -> &mut Self
    where
        R: ?Sized + 'static,
    {
        self.consume_with::<tags::Ref<tags::MaybeSizedValue<R>>>(fulfil)
    }

    fn retrieve_value<V>(&mut self, fulfil: impl FnOnce(V)) -> &mut Self
    where
        V: 'static,
    {
        self.consume_with::<tags::Value<V>>(fulfil)
    }
}
#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a> private::MultiResponseInner<'a> for EmptyMultiResponse {
    #[allow(private_bounds)]
    fn consume_with<I>(&mut self, _fulfil: impl FnOnce(I::Reified)) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        self
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a, J, NEXT> private::MultiResponseInner<'a> for ChainMultiResponse<'a, J, NEXT>
where
    J: tags::Type<'a>,
    NEXT: private::MultiResponseInner<'a>,
{
    fn consume_with<I>(&mut self, fulfil: impl FnOnce(I::Reified)) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        // SAFETY: cast is safe because type ids are equal implies types are equal
        unsafe {
            // this `if` is const. Equality is always decidable for tag types, but we can't prove that to the type system.
            if TypeId::of::<I>() == TypeId::of::<J>() {
                // cast is safe because type ids are equal
                let cur =
                    &mut *(&mut self.cur as *mut Option<J::Reified> as *mut Option<I::Reified>);
                if let Some(val) = cur.take() {
                    fulfil(val);
                    return self;
                }
            }
        }
        self.next.consume_with::<I>(fulfil);
        self
    }
}
#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a, T, NEXT> private::MultiResponseInner<'a> for ChainValMultiResponse<'a, T, NEXT>
where
    T: 'static,
    NEXT: private::MultiResponseInner<'a>,
{
    #[allow(private_bounds)]
    fn consume_with<I>(&mut self, fulfil: impl FnOnce(I::Reified)) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        self.inner.consume_with::<I>(fulfil);
        self
    }
}
#[unstable(feature = "error_generic_member_access", issue = "99301")]
// SAFETY: delegates to inner impl
unsafe impl<'a, T, NEXT> Erased<'a> for ChainValMultiResponse<'a, T, NEXT>
where
    T: 'static,
    NEXT: Erased<'a>,
{
    unsafe fn consume_closure(
        this: impl FnOnce() -> *const Self,
        type_id: TypeId,
    ) -> Option<NonNull<()>> {
        // SAFETY: delegation
        unsafe { ChainMultiResponse::consume_closure(move || &raw const (*this()).inner, type_id) }
    }

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>> {
        // SAFETY: same safety conditions
        unsafe { Self::consume_closure(move || self, type_id) }
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'a, T, NEXT> private::MultiResponseInner<'a> for ChainRefMultiResponse<'a, T, NEXT>
where
    T: 'static + ?Sized,
    NEXT: private::MultiResponseInner<'a>,
{
    #[allow(private_bounds)]
    fn consume_with<I>(&mut self, fulfil: impl FnOnce(I::Reified)) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        self.inner.consume_with::<I>(fulfil);
        self
    }
}
#[unstable(feature = "error_generic_member_access", issue = "99301")]
// SAFETY: delegates to inner impl
unsafe impl<'a, T, NEXT> Erased<'a> for ChainRefMultiResponse<'a, T, NEXT>
where
    T: 'static + ?Sized,
    NEXT: Erased<'a>,
{
    unsafe fn consume_closure(
        this: impl FnOnce() -> *const Self,
        type_id: TypeId,
    ) -> Option<NonNull<()>> {
        // SAFETY: delegation
        unsafe { ChainMultiResponse::consume_closure(move || &raw const (*this()).inner, type_id) }
    }

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>> {
        // SAFETY: same safety conditions
        unsafe { Self::consume_closure(move || self, type_id) }
    }
}

unsafe impl<'a> Erased<'a> for EmptyMultiResponse {
    unsafe fn consume_closure(
        _this: impl FnOnce() -> *const Self,
        _type_id: TypeId,
    ) -> Option<NonNull<()>> {
        None
    }

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>> {
        // SAFETY: same safety conditions
        unsafe { Self::consume_closure(move || self, type_id) }
    }
}

unsafe impl<'a, I, NEXT> Erased<'a> for ChainMultiResponse<'a, I, NEXT>
where
    I: tags::Type<'a>,
    NEXT: Erased<'a>,
{
    unsafe fn consume_closure(
        this: impl FnOnce() -> *const Self,
        type_id: TypeId,
    ) -> Option<NonNull<()>> {
        // SAFETY: dereferencing *this guaranteed to be valid.
        unsafe {
            if type_id == TypeId::of::<I>() {
                // SAFETY: returning an Option<I::Reified> as requested
                Some(
                    NonNull::new_unchecked((&raw const (*this()).cur) as *mut Option<I::Reified>)
                        .cast(),
                )
            } else {
                // SAFETY: safe to delegate consume_closure
                NEXT::consume_closure(move || &raw const (*this()).next, type_id)
            }
        }
    }

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>> {
        // SAFETY: same safety conditions
        unsafe { Self::consume_closure(move || self, type_id) }
    }
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
#[derive(Copy, Clone, Debug)]
/// A [MultiRequestBuilder] is used to request multiple types from an [Error] at once.
///
/// Requesting a type from an [Error] is fairly fast - normally faster than formatting
/// an error - but if you need to request many different error types, it is better
/// to use this API to request them at once.
///
/// # Examples
///
/// ```
/// #![feature(error_generic_member_access)]
/// use core::fmt;
/// use core::error::{Error, MultiResponse, Request};
///
/// #[derive(Debug)]
/// struct MyError {
///     str_field: &'static str,
///     val_field: MyExitCode,
/// }
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// struct MyExitCode(u32);
///
/// impl fmt::Display for MyError {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "Example Error")
///     }
/// }
///
/// impl Error for MyError {
///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
///         request
///             .provide_ref::<str>(self.str_field)
///             .provide_value::<MyExitCode>(self.val_field);
///     }
/// }
///
/// fn main() {
///     let e = MyError {
///         str_field: "hello",
///         val_field: MyExitCode(3),
///     };
///
///     let mut str_val = None;
///     let mut exit_code_val = None;
///     let mut string_val = None;
///     let mut value = core::error::MultiRequestBuilder::new()
///         // request by reference
///         .with_ref::<str>()
///         // and by value
///         .with_value::<MyExitCode>()
///         // and some type that isn't in the error
///         .with_value::<String>()
///         .request(&e)
///         // The error has str by reference
///         .retrieve_ref::<str>(|val| str_val = Some(val))
///         // The error has MyExitCode by value
///         .retrieve_value::<MyExitCode>(|val| exit_code_val = Some(val))
///         // The error does not have a string field, consume will not be called
///         .retrieve_value::<String>(|val| string_val = Some(val));
///
///     assert_eq!(exit_code_val, Some(MyExitCode(3)));
///     assert_eq!(str_val, Some("hello"));
///     assert_eq!(string_val, None);
/// }
/// ```
pub struct MultiRequestBuilder<INNER: IntoMultiRequest> {
    inner: PhantomData<INNER>,
}

impl MultiRequestBuilder<EmptyMultiRequestBuilder> {
    /// Create a new [MultiRequestBuilder]
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn new() -> Self {
        MultiRequestBuilder { inner: PhantomData }
    }
}

impl<INNER: IntoMultiRequest> MultiRequestBuilder<INNER> {
    /// Create a [MultiRequestBuilder] that requests a value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(error_generic_member_access)]
    /// use core::error::{Error, MultiRequestBuilder, MultiResponse};
    ///
    /// fn get_string(e: &dyn Error) -> Option<String> {
    ///     let mut result = None;
    ///     MultiRequestBuilder::new()
    ///         .with_value::<String>()
    ///         .request(e)
    ///         .retrieve_value(|res| result = Some(res));
    ///     result
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn with_value<V: 'static>(
        self,
    ) -> MultiRequestBuilder<ChainValMultiRequestBuilder<V, INNER>> {
        MultiRequestBuilder { inner: PhantomData }
    }

    /// Create a [MultiRequestBuilder] that requests a reference.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(error_generic_member_access)]
    /// use core::error::{Error, MultiRequestBuilder, MultiResponse};
    ///
    /// fn get_string(e: &dyn Error) -> Option<String> {
    ///     let mut result = None;
    ///     MultiRequestBuilder::new()
    ///         .with_value::<String>()
    ///         .request(e)
    ///         .retrieve_value(|res| result = Some(res));
    ///     result
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn with_ref<R: 'static + ?Sized>(
        self,
    ) -> MultiRequestBuilder<ChainRefMultiRequestBuilder<R, INNER>> {
        MultiRequestBuilder { inner: PhantomData }
    }

    /// Request provided values from a given error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request<'a>(self, err: &'a (impl Error + ?Sized)) -> impl MultiResponse<'a> {
        let mut tagged = Tagged::new_virtual(INNER::get_request());
        err.provide(tagged.as_request());
        tagged.value
    }
}

// special type id, used to mark a `Tagged` where calls should be done virtually
struct ErasedMarker;

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

    use crate::any::TypeId;
    use crate::marker::PhantomData;
    use crate::ptr::NonNull;

    /// This trait is implemented by specific tag types in order to allow
    /// describing a type which can be requested for a given lifetime `'a`.
    ///
    /// A few example implementations for type-driven tags can be found in this
    /// module, although crates may also implement their own tags for more
    /// complex types with internal lifetimes.
    pub(crate) unsafe trait Type<'a>: Sized + 'static {
        /// The type of values which may be tagged by this tag for the given
        /// lifetime.
        type Reified: 'a;

        // This requires `sink` to be a valid pointer, and if `type_id == TypeId::of::<T>`` and
        // the function returns Some, returns a pointer with the same lifetime and
        // mutability as `sink` to `Option<<T as Type<'a>>::Reified>`.
        unsafe fn consume(
            sink: *const super::TaggedOption<'a, Self>,
            type_id: TypeId,
        ) -> Option<NonNull<()>>;
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

    unsafe impl<'a, T: 'static> Type<'a> for Value<T> {
        type Reified = T;

        unsafe fn consume(
            sink: *const super::TaggedOption<'a, Self>,
            type_id: TypeId,
        ) -> Option<NonNull<()>> {
            // SAFETY: sink is a valid pointer
            unsafe {
                if (*sink).0.is_none() && type_id == TypeId::of::<Self>() {
                    Some(NonNull::new_unchecked(&raw const (*sink).0 as *mut Self::Reified).cast())
                } else {
                    None
                }
            }
        }
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

    unsafe impl<'a, I: MaybeSizedType<'a>> Type<'a> for Ref<I>
    where
        I::Reified: 'static,
    {
        type Reified = &'a I::Reified;

        unsafe fn consume(
            sink: *const super::TaggedOption<'a, Self>,
            type_id: TypeId,
        ) -> Option<NonNull<()>> {
            // SAFETY: sink is a valid pointer
            unsafe {
                if (*sink).0.is_none() && type_id == TypeId::of::<Self>() {
                    Some(NonNull::new_unchecked(&raw const (*sink).0 as *mut Self::Reified).cast())
                } else {
                    None
                }
            }
        }
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
    fn new_concrete() -> Self {
        Tagged { tag_id: TypeId::of::<I>(), value: TaggedOption::<'a, I>(None) }
    }
}

impl<'a, T: Erased<'a>> Tagged<T> {
    fn new_virtual(value: T) -> Self {
        Tagged { tag_id: TypeId::of::<ErasedMarker>(), value }
    }
}

impl<'a, T: Erased<'a>> Tagged<T> {
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
unsafe trait Erased<'a>: 'a {
    // This requires `self` to be a valid pointer, and if `type_id == TypeId::of::<T>`` and
    // the function returns Some, returns a pointer with the same lifetime and
    // mutability as `self` to `Option<<T as Type<'a>>::Reified>`.

    // in consume_closure, `self = this()`.
    // The optimizer does not do the branch table optimization if your function takes
    // a self pointer, do the closure hack to work around it
    unsafe fn consume_closure(
        this: impl FnOnce() -> *const Self,
        type_id: TypeId,
    ) -> Option<NonNull<()>>
    where
        Self: Sized;

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>>;
}

unsafe impl<'a, I: tags::Type<'a>> Erased<'a> for TaggedOption<'a, I> {
    // This impl is not really used since TaggedOptions are not virtual, but leave it here
    unsafe fn consume_closure(
        this: impl FnOnce() -> *const Self,
        type_id: TypeId,
    ) -> Option<NonNull<()>> {
        // SAFETY: this is a valid pointer
        unsafe { I::consume(&*this(), type_id) }
    }

    unsafe fn consume(self: *const Self, type_id: TypeId) -> Option<NonNull<()>> {
        // SAFETY: same safety conditions
        unsafe { Self::consume_closure(move || self, type_id) }
    }
}

struct Tagged<E: ?Sized> {
    tag_id: TypeId,
    value: E,
}

impl<'a> Tagged<dyn Erased<'a> + 'a> {
    fn is_virtual(&self) -> bool {
        self.tag_id == TypeId::of::<ErasedMarker>()
    }

    #[inline]
    fn would_be_satisfied_by<I>(&self) -> bool
    where
        I: tags::Type<'a>,
    {
        if self.is_virtual() {
            // consume returns None if the space is not satisfied
            // SAFETY: `&raw const self.value` is valid
            unsafe { (&raw const self.value).consume(TypeId::of::<I>()).is_some() }
        } else {
            matches!(self.downcast::<I>(), Some(TaggedOption(None)))
        }
    }

    #[inline]
    fn provide<I>(&mut self, value: I::Reified)
    where
        I: tags::Type<'a>,
    {
        if self.is_virtual() {
            // SAFETY: consume_mut is defined to return either None or Some(I::Reified)
            unsafe {
                if let Some(res) = (&raw const self.value).consume(TypeId::of::<I>()) {
                    let mut ptr: NonNull<Option<I::Reified>> = res.cast();
                    // cast is fine since consume_mut returns a pointer to an Option<I::Reified>
                    // could use `ptr::write` here, but this is not expected to be important enough
                    *ptr.as_mut() = Some(value);
                }
            }
        } else {
            if let Some(res @ TaggedOption(None)) = self.downcast_mut::<I>() {
                res.0 = Some(value);
            }
        }
    }

    #[inline]
    fn provide_with<I>(&mut self, fulfil: impl FnOnce() -> I::Reified)
    where
        I: tags::Type<'a>,
    {
        if self.is_virtual() {
            // SAFETY: consume_mut is defined to return either None or Some(I::Reified)
            unsafe {
                if let Some(res) = (&raw const self.value).consume(TypeId::of::<I>()) {
                    let mut ptr: NonNull<Option<I::Reified>> = res.cast();
                    // cast is fine since consume_mut returns a pointer to an Option<I::Reified>
                    // could use `ptr::write` here, but this is not expected to be important enough
                    *ptr.as_mut() = Some(fulfil());
                }
            }
        } else {
            if let Some(res @ TaggedOption(None)) = self.downcast_mut::<I>() {
                res.0 = Some(fulfil());
            }
        }
    }

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
