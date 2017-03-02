- Start Date: 2014-11-11
- RFC PR: https://github.com/rust-lang/rfcs/pull/461
- Rust Issue: https://github.com/rust-lang/rust/issues/19175

# Summary

Introduce a new thread local storage module to the standard library, `std::tls`,
providing:

* Scoped TLS, a non-owning variant of TLS for any value.
* Owning TLS, an owning, dynamically initialized, dynamically destructed
  variant, similar to `std::local_data` today.

# Motivation

In the past, the standard library's answer to thread local storage was the
`std::local_data` module. This module was designed based on the Rust task model
where a task could be either a 1:1 or M:N task. This design constraint has
[since been lifted][runtime-rfc], allowing for easier solutions to some of the
current drawbacks of the module. While redesigning `std::local_data`, it can
also be scrutinized to see how it holds up to modern-day Rust style, guidelines,
and conventions.

[runtime-rfc]: https://github.com/rust-lang/rfcs/blob/master/text/0230-remove-runtime.md

In general the amount of work being scheduled for 1.0 is being trimmed down as
much as possible, especially new work in the standard library that isn't focused
on cutting back what we're shipping. Thread local storage, however, is such a
critical part of many applications and opens many doors to interesting sets of
functionality that this RFC sees fit to try and wedge it into the schedule. The
current `std::local_data` module simply doesn't meet the requirements of what
one may expect out of a TLS implementation for a language like Rust.

## Current Drawbacks

Today's implementation of thread local storage, `std::local_data`, suffers from
a few drawbacks:

* The implementation is not super speedy, and it is unclear how to enhance the
  existing implementation to be on par with OS-based TLS or `#[thread_local]`
  support. As an example, today a lookup takes `O(log N)` time where N is the
  number of set TLS keys for a task.

  This drawback is also not to be taken lightly. TLS is a fundamental building
  block for rich applications and libraries, and an inefficient implementation
  will only deter usage of an otherwise quite useful construct.

* The types which can be stored into TLS are not maximally flexible. Currently
  only types which ascribe to `'static` can be stored into TLS. It's often the
  case that a type with references needs to be placed into TLS for a short
  period of time, however.

* The interactions between TLS destructors and TLS itself is not currently very
  well specified, and it can easily lead to difficult-to-debug runtime panics or
  undocumented leaks.

* The implementation currently assumes a local `Task` is available. Once the
  runtime removal is complete, this will no longer be a valid assumption.

## Current Strengths

There are, however, a few pros to the usage of the module today which should be
required for any replacement:

* All platforms are supported.
* `std::local_data` allows consuming ownership of data, allowing it to live past
  the current stack frame.

## Building blocks available

There are currently two primary building blocks available to Rust when building
a thread local storage abstraction, `#[thread_local]` and OS-based TLS. Neither
of these are currently used for `std::local_data`, but are generally seen as
"adequately efficient" implementations of TLS. For example, an TLS access of a
`#[thread_local]` global is simply a pointer offset, which when compared to a
`O(log N)` lookup is quite speedy!

With these available, this RFC is motivated in redesigning TLS to make use of
these primitives.

# Detailed design

Three new modules will be added to the standard library:

* The `std::sys::tls` module provides platform-agnostic bindings the OS-based
  TLS support. This support is intended to only be used in otherwise unsafe code
  as it supports getting and setting a `*mut u8` parameter only.

* The `std::tls` module provides a dynamically initialized and dynamically
  destructed variant of TLS. This is very similar to the current
  `std::local_data` module, except that the implicit `Option<T>` is not
  mandated as an initialization expression is required.

* The `std::tls::scoped` module provides a flavor of TLS which can store a
  reference to any type `T` for a scoped set of time. This is a variant of TLS
  not provided today. The backing idea is that if a reference only lives in TLS
  for a fixed set of time then there's no need for TLS to consume ownership of
  the value itself.

  This pattern of TLS is quite common throughout the compiler's own usage of
  `std::local_data` and often more expressive as no dances are required to move
  a value into and out of TLS.

The design described below can be found as an existing cargo package:
https://github.com/alexcrichton/tls-rs.

## The OS layer

While LLVM has support for `#[thread_local]` statics, this feature is not
supported on all platforms that LLVM can target. Almost all platforms, however,
provide some form of OS-based TLS. For example Unix normally comes with
`pthread_key_create` while Windows comes with `TlsAlloc`.

This RFC proposes introducing a `std::sys::tls` module which contains bindings
to the OS-based TLS mechanism. This corresponds to the `os` module in the
example implementation. While not currently public, the contents of `sys` are
slated to become public over time, and the API of the `std::sys::tls` module
will go under API stabilization at that time.

This module will support "statically allocated" keys as well as dynamically
allocated keys. A statically allocated key will actually allocate a key on
first use.

### Destructor support

The major difference between Unix and Windows TLS support is that Unix supports
a destructor function for each TLS slot while Windows does not. When each Unix
TLS key is created, an optional destructor is specified. If any key has a
non-NULL value when a thread exits, the destructor is then run on that value.

One possibility for this `std::sys::tls` module would be to not provide
destructor support at all (least common denominator), but this RFC proposes
implementing destructor support for Windows to ensure that functionality is not
lost when writing Unix-only code.

Destructor support for Windows will be provided through a custom implementation
of tracking known destructors for TLS keys.

## Scoped TLS

As discussed before, one of the motivations for this RFC is to provide a method
of inserting any value into TLS, not just those that ascribe to `'static`. This
provides maximal flexibility in storing values into TLS to ensure any "thread
local" pattern can be encompassed.

Values which do not adhere to `'static` contain references with a constrained
lifetime, and can therefore not be moved into TLS. They can, however, be
*borrowed* by TLS. This scoped TLS api provides the ability to insert a
reference for a particular period of time, and then a non-escaping reference can
be extracted at any time later on.

In order to implement this form of TLS, a new module, `std::tls::scoped`, will
be added. It will be coupled with a `scoped_tls!` macro in the prelude. The API
looks like:

```rust
/// Declares a new scoped TLS key. The keyword `static` is required in front to
/// emphasize that a `static` item is being created. There is no initializer
/// expression because this key initially contains no value.
///
/// A `pub` variant is also provided to generate a public `static` item.
macro_rules! scoped_tls(
    (static $name:ident: $t:ty) => (/* ... */);
    (pub static $name:ident: $t:ty) => (/* ... */);
)

/// A structure representing a scoped TLS key.
///
/// This structure cannot be created dynamically, and it is accessed via its
/// methods.
pub struct Key<T> { /* ... */ }

impl<T> Key<T> {
    /// Insert a value into this scoped TLS slot for a duration of a closure.
    ///
    /// While `cb` is running, the value `t` will be returned by `get` unless
    /// this function is called recursively inside of cb.
    ///
    /// Upon return, this function will restore the previous TLS value, if any
    /// was available.
    pub fn set<R>(&'static self, t: &T, cb: || -> R) -> R { /* ... */ }

    /// Get a value out of this scoped TLS variable.
    ///
    /// This function takes a closure which receives the value of this TLS
    /// variable, if any is available. If this variable has not yet been set,
    /// then None is yielded.
    pub fn with<R>(&'static self, cb: |Option<&T>| -> R) -> R { /* ... */ }
}
```

The purpose of this module is to enable the ability to insert a value into TLS
for a scoped period of time. While able to cover many TLS patterns, this flavor
of TLS is not comprehensive, motivating the owning variant of TLS.

### Variations

Specifically the `with` API can be somewhat unwieldy to use. The `with` function
takes a closure to run, yielding a value to the closure.  It is believed that
this is required for the implementation to be sound, but it also goes against
the "use RAII everywhere" principle found elsewhere in the stdlib.

Additionally, the `with` function is more commonly called `get` for accessing a
contained value in the stdlib. The name `with` is recommended because it may be
possible in the future to express a `get` function returning a reference with a
lifetime bound to the stack frame of the caller, but it is not currently
possible to do so.

The `with` functions yields an `Option<&T>` instead of `&T`. This is to cover
the use case where the key has not been `set` before it used via `with`. This is
somewhat unergonomic, however, as it will almost always be followed by
`unwrap()`. An alternative design would be to provide a `is_set` function and
have `with` `panic!` instead.

## Owning TLS

Although scoped TLS can store any value, it is also limited in the fact that it
cannot own a value. This means that TLS values cannot escape the stack from from
which they originated from. This is itself another common usage pattern of TLS,
and to solve this problem the `std::tls` module will provided support for
placing owned values into TLS.

These values must not contain references as that could trigger a use-after-free,
but otherwise there are no restrictions on placing statics into owned TLS. The
module will support dynamic initialization (run on first use of the variable) as
well as dynamic destruction (implementors of `Drop`).

The interface provided will be similar to what `std::local_data` provides today,
except that the `replace` function has no analog (it would be written with a
`RefCell<Option<T>>`).

```rust
/// Similar to the `scoped_tls!` macro, except allows for an initializer
/// expression as well.
macro_rules! tls(
    (static $name:ident: $t:ty = $init:expr) => (/* ... */)
    (pub static $name:ident: $t:ty = $init:expr) => (/* ... */)
)

pub struct Key<T: 'static> { /* ... */ }

impl<T: 'static> Key<T> {
    /// Access this TLS variable, lazily initializing it if necessary.
    ///
    /// The first time this function is called on each thread the TLS key will
    /// be initialized by having the specified init expression evaluated on the
    /// current thread.
    ///
    /// This function can return `None` for the same reasons of static TLS
    /// returning `None` (destructors are running or may have run).
    pub fn with<R>(&'static self, f: |Option<&T>| -> R) -> R { /* ... */ }
}
```

### Destructors

One of the major points about this implementation is that it allows for values
with destructors, meaning that destructors must be run when a thread exits. This
is similar to placing a value with a destructor into `std::local_data`. This RFC
attempts to refine the story around destructors:

* A TLS key cannot be accessed while its destructor is running. This is
  currently manifested with the `Option` return value.
* A TLS key *may* not be accessible after its destructor has run.
* Re-initializing TLS keys during destruction may cause memory leaks (e.g.
  setting the key FOO during the destructor of BAR, and initializing BAR in the
  destructor of FOO). An implementation will strive to destruct initialized
  keys whenever possible, but it may also result in a memory leak.
* A `panic!` in a TLS destructor will result in a process abort. This is similar
  to a double-failure.

These semantics are still a little unclear, and the final behavior may still
need some more hammering out. The sample implementation suffers from a few extra
drawbacks, but it is believed that some more implementation work can overcome
some of the minor downsides.

### Variations

Like the scoped TLS variation, this key has a `with` function instead of the
normally expected `get` function (returning a reference). One possible
alternative would be to yield `&T` instead of `Option<&T>` and `panic!` if the
variable has been destroyed. Another possible alternative is to have a `get`
function returning a `Ref<T>`. Currently this is unsafe, however, as there is no
way to ensure that `Ref<T>` does not satisfy `'static`. If the returned
reference satisfies `'static`, then it's possible for TLS values to reference
each other after one has been destroyed, causing a use-after-free.

# Drawbacks

* There is no variant of TLS for statically initialized data. Currently the
  `std::tls` module requires dynamic initialization, which means a slight
  penalty is paid on each access (a check to see if it's already initialized).
* The specification of destructors on owned TLS values is still somewhat shaky
  at best. It's possible to leak resources in unsafe code, and it's also
  possible to have different behavior across platforms.
* Due to the usage of macros for initialization, all fields of `Key` in all
  scenarios must be public. Note that `os` is excepted because its initializers
  are a `const`.
* This implementation, while declared safe, is not safe for systems that do any
  form of multiplexing of many threads onto one thread (aka green tasks or
  greenlets). This RFC considers it the multiplexing systems' responsibility to
  maintain native TLS if necessary, or otherwise strongly recommend not using
  native TLS.

# Alternatives

Alternatives on the API can be found in the "Variations" sections above.

Some other alternatives might include:

* A 0-cost abstraction over `#[thread_local]` and OS-based TLS which does not
  have support for destructors but requires static initialization. Note that
  this variant still needs destructor support *somehow* because OS-based TLS
  values must be pointer-sized, implying that the rust value must itself be
  boxed (whereas `#[thread_local]` can support any type of any size).

* A variant of the `tls!` macro could be used where dynamic initialization is
  opted out of because it is not necessary for a particular use case.

* A [previous PR][prev-pr] from @thestinger leveraged macros more heavily than
  this RFC and provided statically constructible Cell and RefCell equivalents
  via the usage of `transmute`. The implementation provided did not, however,
  include the scoped form of this RFC.

[prev-pr]: https://github.com/rust-lang/rust/pull/17583

# Unresolved questions

* Are the questions around destructors vague enough to warrant the `get` method
  being `unsafe` on owning TLS?
* Should the APIs favor `panic!`-ing internally, or exposing an `Option`?
