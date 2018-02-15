- Feature Name: `track_caller`
- Start Date: 2017-07-31
- RFC PR: [rust-lang/rfcs#2091](https://github.com/rust-lang/rfcs/pull/2091)
- Rust Issue: [rust-lang/rust#47809](https://github.com/rust-lang/rust/issues/47809)

----

# Summary
[summary]: #summary

Enable accurate caller location reporting during panic in `{Option, Result}::{unwrap, expect}` with
the following changes:

1. Support the `#[track_caller]` function attribute, which guarantees a function has access to the
    caller information.
2. Add an intrinsic function `caller_location()` (safe wrapper: `Location::caller()`) to retrieve
    the caller's source location.

Example:

```rust
#![feature(track_caller)]
use std::panic::Location;

#[track_caller]
fn unwrap(self) -> T {
    panic!("{}: oh no", Location::caller());
}

let n: Option<u32> = None;
let m = n.unwrap();
```

<!-- TOC updateOnSave:false -->

- [Summary](#summary)
- [Motivation](#motivation)
- [Guide-level explanation](#guide-level-explanation)
    - [Let's reimplement `unwrap()`](#lets-reimplement-unwrap)
    - [Track the caller](#track-the-caller)
    - [Location type](#location-type)
    - [Propagation of tracker](#propagation-of-tracker)
    - [Why do we use implicit caller location](#why-do-we-use-implicit-caller-location)
- [Reference-level explanation](#reference-level-explanation)
    - [Survey of panicking standard functions](#survey-of-panicking-standard-functions)
    - [Procedural attribute macro](#procedural-attribute-macro)
    - [Redirection (MIR inlining)](#redirection-mir-inlining)
    - [Standard libraries](#standard-libraries)
    - [“My fault” vs “Your fault”](#my-fault-vs-your-fault)
    - [Location detail control](#location-detail-control)
- [Drawbacks](#drawbacks)
    - [Code bloat](#code-bloat)
    - [Narrow solution scope](#narrow-solution-scope)
    - [Confusing scoping rule](#confusing-scoping-rule)
- [Rationale and alternatives](#rationale-and-alternatives)
    - [Rationale](#rationale)
    - [Alternatives](#alternatives)
        - [🚲 Name of everything 🚲](#-name-of-everything-)
        - [Using an ABI instead of an attribute](#using-an-abi-instead-of-an-attribute)
        - [Repurposing `file!()`, `line!()`, `column!()`](#repurposing-file-line-column)
        - [Inline MIR](#inline-mir)
        - [Default function arguments](#default-function-arguments)
        - [Semantic inlining](#semantic-inlining)
        - [Design-by-contract](#design-by-contract)
    - [Non-viable alternatives](#non-viable-alternatives)
        - [Macros](#macros)
        - [Backtrace](#backtrace)
        - [`SourceContext` generic parameter](#sourcecontext-generic-parameter)
- [Unresolved questions](#unresolved-questions)

<!-- /TOC -->

# Motivation
[motivation]: #motivation

It is well-known that the error message reported by `unwrap()` is useless:

```text
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', /checkout/src/libcore/option.rs:335
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```

There have been numerous discussions ([a], [b], [c]) that want `unwrap()` and friends to provide
better information to locate the panic. [RFC 1669] attempted to address this by
introducing the `unwrap!(x)` macro to the standard library, but it was closed since the `x.unwrap()`
convention is too entrenched.

This RFC introduces line numbers into `unwrap()` without requiring users to adapt a new
idiom, i.e. the user should be able to see the precise location without changing any source
code.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Let's reimplement `unwrap()`

`unwrap()` and `expect()` are two methods on `Option` and `Result` that are commonly used when you
are *absolutely sure* they contain a successful value and you want to extract it.

```rust
// 1.rs
use std::env::args;
fn main() {
    println!("args[1] = {}", args().nth(1).unwrap());
    println!("args[2] = {}", args().nth(2).unwrap());
    println!("args[3] = {}", args().nth(3).unwrap());
}
```

If the assumption is wrong, they will panic and tell you that an error is unexpected.

```text
$ ./1
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', 1.rs:4:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./1 arg1
args[1] = arg1
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', 1.rs:5:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./1 arg1 arg2
args[1] = arg1
args[2] = arg2
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', 1.rs:6:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./1 arg1 arg2 arg3
args[1] = arg1
args[2] = arg2
args[3] = arg3
```

Let's say you are unhappy with these built-in functions, e.g. you want to provide an alternative
error message:

```rust
// 2.rs
use std::env::args;
pub fn my_unwrap<T>(input: Option<T>) -> T {
    match input {
        Some(t) => t,
        None => panic!("nothing to see here, move along"),
    }
}
fn main() {
    println!("args[1] = {}", my_unwrap(args().nth(1)));
    println!("args[2] = {}", my_unwrap(args().nth(2)));
    println!("args[3] = {}", my_unwrap(args().nth(3)));
}
```

This trivial implementation, however, will only report the panic that happens inside `my_unwrap`. This is
pretty useless since it is the caller of `my_unwrap` that made the wrong assumption!

```text
$ ./2
thread 'main' panicked at 'nothing to see here, move along', 2.rs:5:16
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./2 arg1
args[1] = arg1
thread 'main' panicked at 'nothing to see here, move along', 2.rs:5:16
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./2 arg1 arg2
args[1] = arg1
args[2] = arg2
thread 'main' panicked at 'nothing to see here, move along', 2.rs:5:16
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./2 arg1 arg2 arg3
args[1] = arg1
args[2] = arg2
args[3] = arg3
```

The trivial solution would require the user to provide `file!()`, `line!()` and `column!()`. A
slightly more ergonomic solution would be changing `my_unwrap` into a macro, allowing these constants to
be automatically provided.

```rust
pub fn my_unwrap_at_source_location<T>(input: Option<T>, file: &str, line: u32, column: u32) -> T {
    match input {
        Some(t) => t,
        None => panic!("nothing to see at {}:{}:{}, move along", file, line, column),
    }
}

macro_rules! my_unwrap {
    ($input:expr) => {
        my_unwrap_at_source_location($input, file!(), line!(), column!())
    }
}
println!("args[1] = {}", my_unwrap!(args().nth(1)));
//                                ^ tell user to add an `!`.
...
```

What if you have already published the `my_unwrap` crate that has thousands of users, and you
want to maintain API stability? Before Rust 1.XX, the builtin `unwrap()` had the same problem!

## Track the caller

The reason the `my_unwrap!` macro works is because it copy-and-pastes the entire content of its macro
definition every time it is used.

```rust
println!("args[1] = {}", my_unwrap!(args().nth(1)));
println!("args[2] = {}", my_unwrap!(args().nth(2)));
...

// is equivalent to:

println!("args[1] = {}", my_unwrap(args().nth(1), file!(), line!(), column!()));
println!("args[1] = {}", my_unwrap(args().nth(2), file!(), line!(), column!()));
...
```

What if we could instruct the compiler to automatically fill in the file, line, and column?
Rust 1.YY introduced the `#[track_caller]` attribute for exactly this reason:

```rust
// 3.rs
#![feature(track_caller)]
use std::env::args;
#[track_caller]  // <-- Just add this!
pub fn my_unwrap<T>(input: Option<T>) -> T {
    match input {
        Some(t) => t,
        None => panic!("nothing to see here, move along"),
    }
}
fn main() {
    println!("args[1] = {}", my_unwrap(args().nth(1)));
    println!("args[2] = {}", my_unwrap(args().nth(2)));
    println!("args[3] = {}", my_unwrap(args().nth(3)));
}
```

Now we have truly reproduced how the built-in `unwrap()` is implemented.

```text
$ ./3
thread 'main' panicked at 'nothing to see here, move along', 3.rs:12:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./3 arg1
args[1] = arg1
thread 'main' panicked at 'nothing to see here, move along', 3.rs:13:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./3 arg1 arg2
args[1] = arg1
args[2] = arg2
thread 'main' panicked at 'nothing to see here, move along', 3.rs:14:29
note: Run with `RUST_BACKTRACE=1` for a backtrace.

$ ./3 arg1 arg2 arg3
args[1] = arg1
args[2] = arg2
args[3] = arg3
```

`#[track_caller]` is an automated version of what you've seen in the last section. The attribute
copies `my_unwrap` to a new function `my_unwrap_at_source_location` which accepts the caller's
location as an additional argument. The attribute also instructs the compiler to replace
`my_unwrap(x)` with `my_unwrap_at_source_location(x, file!(), line!(), column!())` (sort of)
whenever it sees it. This allows us to maintain the stability guarantee while allowing the user to
get the new behavior with just one recompile.

## Location type

Let's enhance `my_unwrap` to also log a message to the log file before panicking. We would need to
get the caller's location as a value. This is supported using the method `Location::caller()`:

```rust
use std::panic::Location;
#[track_caller]
pub fn my_unwrap<T>(input: Option<T>) -> T {
    match input {
        Some(t) => t,
        None => {
            let location = Location::caller();
            println!("unwrapping a None from {}:{}", location.file(), location.line());
            panic!("nothing to see here, move along")
        }
    }
}
```

## Propagation of tracker

When your `#[track_caller]` function calls another `#[track_caller]` function, the caller location
will be propagated downwards:

```rust
use std::panic::Location;
#[track_caller]
pub fn my_get_index<T>(input: &[T], index: usize) -> &T {
    my_unwrap(input.get(index))        // line 4
}
indirectly_unwrap(None);    // line 6
```

When you run this, the panic will refer to line 6, the original caller, instead of line 4 where
`my_get_index` calls `my_unwrap`. When a library function is marked `#[track_caller]`, it is
expected the function is short, and does not have any logic errors. This allows us to always track
the caller on failure.

If a panic that refers to the local location is actually needed, you may workaround by wrapping the
code in a closure which cannot track the caller:

```rust
#[track_caller]
pub fn my_get_index<T>(input: &[T], index: usize) -> &T {
    (|| {
        my_unwrap(input.get(index))
    })()
}
```

## Why do we use implicit caller location

If you are learning Rust alongside other languages, you may wonder why Rust obtains the caller
information in such a strange way. There are two restrictions that force us to adopt this solution:

1. Programmatic access to the stack backtrace is often used in interpreted or runtime-heavy
    languages like Python and Java. However, the stack backtrace is not suitable as the only
    solution for systems languages like Rust because optimization often collapses multiple levels
    of function calls.  In some embedded systems, the backtrace may even be unavailable!

2. Solutions that use default function arguments alongside normal arguments are are often used in
    languages that do not perform inference higher than statement level, e.g. Swift and C#. Rust
    does not (yet) support default function arguments or function overloading because they interfere
    with type inference, so such solutions are ruled out.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Survey of panicking standard functions

Many standard functions may panic. These are divided into three categories depending on whether they
should receive caller information despite the inlining cost associated with it.

The list of functions is not exhaustive. Only those with a "Panics" section in the documentation
are included.

1. **Must have.** These functions are designed to generate a panic, or used so often that indicating
    a panic happening from them often gives no useful information.

    | Function | Panic condition |
    |:---------|:----------------|
    | `Option::expect` | self is None |
    | `Option::unwrap` | self is None |
    | `Result::expect_err` | self is Ok |
    | `Result::expect` | self is Err |
    | `Result::unwrap_err` | self is Ok |
    | `Result::unwrap` | self is Err |
    | `[T]::index_mut` | range out of bounds |
    | `[T]::index` | range out of bounds |
    | `BTreeMap::index` | key not found |
    | `HashMap::index` | key not found |
    | `str::index_mut` | range out of bounds or off char boundary |
    | `str::index` | range out of bounds or off char boundary |
    | `VecDeque::index_mut` | index out of bounds |
    | `VecDeque::index` | index out of bounds |

2. **Nice to have.** These functions are not commonly used, or the panicking condition is pretty
    rare. Often the panic information contains enough clue to fix the error without a backtrace.
    Inlining them would bloat the binary size without much benefit.

    <details><summary>List of category 2 functions</summary>

    | Function | Panic condition |
    |:---------|:----------------|
    | `std::env::args` | non UTF-8 values |
    | `std::env::set_var` | invalid key or value |
    | `std::env::vars` | non UTF-8 values |
    | `std::thread::spawn` | OS failed to create the thread |
    | `[T]::clone_from_slice` | slice lengths differ |
    | `[T]::copy_from_slice` | slice lengths differ |
    | `[T]::rotate` | index out of bounds |
    | `[T]::split_at_mut` | index out of bounds |
    | `[T]::swap` | index out of bounds
    | `BinaryHeap::reserve_exact` | capacity overflow |
    | `BinaryHeap::reserve` | capacity overflow |
    | `Duration::new` | arithmetic overflow |
    | `HashMap::reserve` | capacity overflow |
    | `HashSet::reserve` | capacity overflow |
    | `i32::overflowing_div` | zero divisor |
    | `i32::overflowing_rem` | zero divisor |
    | `i32::wrapping_div` | zero divisor |
    | `i32::wrapping_rem` | zero divisor |
    | `Instance::duration_since` | time travel |
    | `Instance::elapsed` | time travel |
    | `Iterator::count` | extremely long iterator |
    | `Iterator::enumerate` | extremely long iterator |
    | `Iterator::position` | extremely long iterator |
    | `Iterator::product` | arithmetic overflow in debug build |
    | `Iterator::sum` | arithmetic overflow in debug build |
    | `LinkedList::split_off` | index out of bounds |
    | `LocalKey::with` | TLS has been destroyed |
    | `RawVec::double_in_place` | capacity overflow |
    | `RawVec::double` | capacity overflow |
    | `RawVec::reserve_exact` | capacity overflow |
    | `RawVec::reserve_in_place` | capacity overflow |
    | `RawVec::reserve` | capacity overflow |
    | `RawVec::shrink_to_fit` | given amount is larger than current capacity |
    | `RawVec::with_capacity` | capacity overflow |
    | `RefCell::borrow_mut` | a borrow or mutable borrow is active |
    | `RefCell::borrow` | a mutable borrow is active |
    | `str::split_at_mut` | range out of bounds or off char boundary |
    | `str::split_at` | range out of bounds or off char boundary |
    | `String::drain` | range out of bounds or off char boundary |
    | `String::insert_str` | index out of bounds or off char boundary |
    | `String::insert` | index out of bounds or off char boundary |
    | `String::remove` | index out of bounds or off char boundary |
    | `String::reserve_exact` | capacity overflow |
    | `String::reserve` | capacity overflow |
    | `String::splice` | range out of bounds or off char boundary |
    | `String::split_off` | index out of bounds or off char boundary |
    | `String::truncate` | off char boundary |
    | `Vec::append` | capacity overflow |
    | `Vec::drain` | range out of bounds |
    | `Vec::insert` | index out of bounds |
    | `Vec::push` | capacity overflow |
    | `Vec::remove` | index out of bounds |
    | `Vec::reserve_exact` | capacity overflow |
    | `Vec::reserve` | capacity overflow |
    | `Vec::splice` | range out of bounds |
    | `Vec::split_off` | index out of bounds |
    | `Vec::swap_remove` | index out of bounds |
    | `VecDeque::append` | capacity overflow |
    | `VecDeque::drain` | range out of bounds |
    | `VecDeque::insert` | index out of bounds |
    | `VecDeque::reserve_exact` | capacity overflow |
    | `VecDeque::reserve` | capacity overflow |
    | `VecDeque::split_off` | index out of bounds |
    | `VecDeque::swap` | index out of bounds |
    | `VecDeque::with_capacity` | capacity overflow |

    </details>

3. **Not needed.** Panics from these indicate silly programmer error and the panic itself has
    enough clue to let programmers figure out where the error comes from.

    <details><summary>List of category 3 functions</summary>

    | Function | Panic condition |
    |:---------|:----------------|
    | `std::atomic::fence` | using invalid atomic ordering |
    | `std::char::from_digit` | radix is outside `2 ..= 36` |
    | `std::env::remove_var` | invalid key |
    | `std::format!` | the `fmt` method returns Err |
    | `std::panicking::set_hook` | called in panicking thread |
    | `std::panicking::take_hook` | called in panicking thread |
    | `[T]::chunks_mut` | chunk size == 0 |
    | `[T]::chunks` | chunk size == 0 |
    | `[T]::windows` | window size == 0 |
    | `AtomicUsize::compare_exchange_weak` | using invalid atomic ordering |
    | `AtomicUsize::compare_exchange` | using invalid atomic ordering |
    | `AtomicUsize::load` | using invalid atomic ordering |
    | `AtomicUsize::store` | using invalid atomic ordering |
    | `BorrowRef::clone` | borrow counter overflows, see [issue 33880] |
    | `BTreeMap::range_mut` | end of range before start of range |
    | `BTreeMap::range` | end of range before start of range |
    | `char::encode_utf16` | dst buffer smaller than `[u16; 2]` |
    | `char::encode_utf8` | dst buffer smaller than `[u8; 4]` |
    | `char::is_digit` | radix is outside `2 ..= 36` |
    | `char::to_digit` | radix is outside `2 ..= 36` |
    | `compiler_fence` | using invalid atomic ordering |
    | `Condvar::wait` | waiting on multiple different mutexes |
    | `Display::to_string` | the `fmt` method returns Err |
    | `ExactSizeIterator::len` | size_hint implemented incorrectly |
    | `i32::from_str_radix` | radix is outside `2 ..= 36` |
    | `Iterator::step_by` | step == 0 |

    </details>

This RFC only advocates adding the `#[track_caller]` attribute to the `unwrap` and `expect`
functions. The `index` and `index_mut` functions should also have it if possible, but this is
currently postponed as it is not investigated yet how to insert the transformation after
monomorphization.

## Procedural attribute macro

The `#[track_caller]` attribute will modify a function at the AST and MIR levels without touching
the type-checking (HIR level) or the low-level LLVM passes.

It will first wrap the body of the function in a closure, and then call it:

```rust
#[track_caller]
fn foo<C>(x: A, y: B, z: C) -> R {
    bar(x, y)
}

// will become:

#[rustc_implicit_caller_location]
#[inline]
fn foo<C>(x: A, y: B, z: C) -> R {
    std::ops::FnOnce::call_once(move |__location| {
        bar(x, y)
    }, (unsafe { std::intrinsics::caller_location() },))
}
```

This is to split the function into two: the function `foo` itself, and the closure
`foo::{{closure}}` in it. (Technically: it is the simplest way to create two `DefId`s at the HIR
level as far as I know.)

The function signature of `foo` remains unchanged, so typechecking can proceed normally. The
attribute will be replaced by `#[rustc_implicit_caller_location]` to let the compiler internals
continue to treat it specially. `#[inline]` is added so external crates can see through `foo` to
find `foo::{{closure}}`.

The closure `foo::{{closure}}` is a proper function so that the compiler can write calls directly to
`foo::{{closure}}`, skipping `foo`. Multiple calls to `foo` from different locations can be done via
calling `foo::{{closure}}` directly, instead of copying the function body every time which would
bloat the binary size.

The intrinsic `caller_location()` is a placeholder which will be replaced by the actual caller
location when one calls `foo::{{closure}}` directly.

Currently the `foo::{{closure}}` cannot inherit attributes defined on the main function. To prevent
problems regarding ABI, using `#[naked]` or `extern "ABI"` together with
`#[rustc_implicit_caller_location]` should raise an error.

## Redirection (MIR inlining)

After all type-checking and validation is done, we can now inject the caller location. This is done
by redirecting all calls to `foo` to `foo::{{closure}}`.

```rust
_r = call foo(_1, _2, _3) -> 'bb1;

// will become:

_c = call std::intrinsics::caller_location() -> 'bbt;
'bbt:
_r = call foo::{{closure}} (&[closure: x: _1, y: _2], _c) -> 'bb1;
```

We will further replace the `caller_location()` intrinsic according to where `foo` is called.
If it is called from an ordinary function, it would be replaced by the callsite's location:

```rust
// for ordinary functions,

_c = call std::intrinsics::caller_location() -> 'bbt;

// will become:

_c = Location { file: file!(), line: line!(), column: column!() };
goto -> 'bbt;
```

If it is called from an `#[rustc_implicit_caller_location]`'s closure e.g. `foo::{{closure}}`, the
intrinsic will be replaced by the closure argument `__location` instead, so that the caller location
can propagate directly

```rust
// for #[rustc_implicit_caller_location] closures,

_c = call std::intrinsics::caller_location() -> 'bbt;

// will become:

_c = __location;
goto -> 'bbt;
```

These steps are very similar to inlining, and thus the first proof-of-concept is implemented
directly as a variant of the MIR inliner (but a separate pass). This also means the redirection pass
currently suffers from all disadvantages of the MIR inliner, namely:

* Locations will not be propagated into diverging functions (`fn() -> !`), since inlining them is
    not supported yet.

* MIR passes are run *before* monomorphization, meaning `#[track_caller]` currently **cannot** be
    used on trait items:

```rust
trait Trait {
    fn unwrap(&self);
}
impl Trait for u64 {
    #[track_caller] //~ ERROR: `#[track_caller]` is not supported for trait items yet.
    fn unwrap(&self) {}
}
```

To support trait items, the redirection pass must be run as post-monomorphized MIR pass (which does
not exist yet), or converted to queries provided after resolve, or a custom LLVM inlining pass which
can extract the caller's source location. This prevents the `Index` trait from having
`#[track_caller]` yet.

We cannot hack the impl resolution method into pre-monomorphization MIR pass because of deeply
nested functions like

```rust
f1::<u32>();

fn f1<T: Trait>() { f2::<T>(); }
fn f2<T: Trait>() { f3::<T>(); }
fn f3<T: Trait>() { f4::<T>(); }
...
fn f100<T: Trait>() {
    T::unwrap(); // No one will know T is u32 before monomophization.
}
```

Currently the redirection pass always runs before the inlining pass. If the redirection pass is run
after the normal MIR inlining pass, the normal MIR inliner must treat
`#[rustc_implicit_caller_location]` as `#[inline(never)]`.

The closure `foo::{{closure}}` must never be inlined before the redirection pass.

When `#[rustc_implicit_caller_location]` functions are called dynamically, no inlining will occur,
and thus it cannot take the location of the caller. Currently this will report where the function is
declared. Taking the address of such functions must be allowed due to backward compatibility. (If
a post-monomorphized MIR pass exists, methods via trait objects would be another case of calling
`#[rustc_implicit_caller_location]` functions without caller location.)

```rust
let f: fn(Option<u32>) -> u32 = Option::unwrap;
let g: fn(Option<u32>) -> u32 = Option::unwrap;
assert!(f == g); // This must remain `true`.
f(None);
g(None); // The effect of these two calls must be the same.
```

## Standard libraries

The `caller_location()` intrinsic returns the `Location` structure which encodes the file, line and
column of the callsite. This shares the same structure as the existing type `std::panic::Location`.
Therefore, the type is promoted to a lang-item, and moved into `core::panicking::Location`. It is
re-exported from `libstd`.

Thanks to how `#[track_caller]` is implemented, we could provide a safe wrapper around the
`caller_location()` intrinsic:

```rust
impl<'a> Location<'a> {
    #[track_caller]
    pub fn caller() -> Location<'static> {
        unsafe {
            ::intrinsics::caller_location()
        }
    }
}
```

The `panic!` macro is modified to use `Location::caller()` (or the intrinsic directly) so it can
report the caller location inside `#[track_caller]`.

```rust
macro_rules! panic {
    ($msg:expr) => {
        let loc = $crate::panicking::Location::caller();
        $crate::panicking::panic(&($msg, loc.file(), loc.line(), loc.column()))
    };
    ...
}
```

Actually this is now more natural for `core::panicking::panic_fmt` to take `Location` directly
instead of tuples, so one should consider changing their signature, but this is out-of-scope for
this RFC.

`panic!` is often used outside of `#[track_caller]` functions. In those cases, the
`caller_location()` intrinsic will pass unchanged through all MIR passes into trans. As a fallback,
the intrinsic will expand to `Location { file: file!(), line: line!(), col: column!() }` during
trans.

## “My fault” vs “Your fault”

In a `#[track_caller]` function, we expect all panics being attributed to the caller (thus the
attribute name). However, sometimes the code panics not due to the caller, but the implementation
itself. It may be important to distinguish between "my fault" (implementation error) and
"your fault" (caller violating API requirement). As an example,

```rust
use std::collections::HashMap;
use std::hash::Hash;

fn count_slices<T: Hash + Eq>(array: &[T], window: usize) -> HashMap<&[T], usize> {
    if !(0 < window && window <= array.len()) {
        panic!("invalid window size");
        // ^ triggering this panic is "your fault"
    }
    let mut result = HashMap::new();
    for w in array.windows(window) {
        if let Some(r) = result.get_mut(w) {
            *r += 1;
        } else {
            panic!("why??");
            // ^ triggering this panic is "my fault"
            //   (yes this code is wrong and entry API should be used)
        }
    }

    result
}
```

One simple solution is to separate the "my fault" panic and "your fault" panic into two, but since
[declarative macro 1.0 is insta-stable][insta-stable], this RFC would prefer to postpone introducing
any new public macros until "Macros 2.0" lands, where stability and scoping are better handled.

For comparison, the Swift language does
[distinguish between the two kinds of panics semantically][swift-panics]. The "your fault" ones are
called `precondition`, while the "my fault" ones are called `assert`, though they don't deal with
caller location, and practically they are equivalent to Rust's `assert!` and `debug_assert!`.
Nevertheless, this also suggests we can still separate existing panicking macros into the "my fault"
and "your fault" camps accordingly:
* Definitely "my fault" (use actual location): `debug_assert!` and friends, `unreachable!`,
    `unimplemented!`
* Probably "your fault" (propagate caller location): `assert!` and friends, `panic!`

The question is, should calling `unwrap()`, `expect()` and `x[y]` (`index()`) be "my fault" or "your
fault"? Let's consider existing implementation of `index()` methods:
```rust
// Vec::index
fn index(&self, index: usize) -> &T {
    &(**self)[index]
}

// BTreeMap::index
fn index(&self, key: &Q) -> &V {
    self.get(key).expect("no entry found for key")
}

// Wtf8::index
fn index(&self, range: ops::RangeFrom<usize>) -> &Wtf8 {
    // is_code_point_boundary checks that the index is in [0, .len()]
    if is_code_point_boundary(self, range.start) {
        unsafe { slice_unchecked(self, range.start, self.len()) }
    } else {
        slice_error_fail(self, range.start, self.len())
    }
}
```

If they all get `#[track_caller]`, the `x[y]`, `expect()` and `slice_error_fail()` should all report
"your fault", i.e. caller location should be propagated downstream. It does mean that the current
default of caller-location-propagation-by-default is more common. This also means "my fault"
happening during development may become harder to spot. This can be solved using `RUST_BACKTRACE=1`,
or workaround by splitting into two functions:

```rust
use std::collections::HashMap;
use std::hash::Hash;

#[track_caller]
fn count_slices<T: Hash + Eq>(array: &[T], window: usize) -> HashMap<&[T], usize> {
    if !(0 < window && window <= array.len()) {
        panic!("invalid window size");  // <-- your fault
    }
    (|| {
        let mut result = HashMap::new();
        for w in array.windows(window) {
            if let Some(r) = result.get_mut(w) {
                *r += 1;
            } else {
                panic!("why??"); // <-- my fault (caller propagation can't go into closures)
            }
        }
        result
    })()
}
```

Anyway, treating everything as "your fault" will encourage that `#[track_caller]` functions should
be short, which goes in line with the ["must have" list](#survey-of-panicking-standard-functions) in
the RFC. Thus the RFC will remain advocating for propagating caller location implicitly.

[insta-stable]: https://github.com/rust-lang/rust/pull/39229#issuecomment-274348420
[swift-panics]: https://stackoverflow.com/questions/29673027/difference-between-precondition-and-assert-in-swift

## Location detail control

An unstable flag `-Z location-detail` is added to `rustc` to control how much factual detail will
be emitted when using `caller_location()`. The user can toggle `file`, `line` and `column` separately,
e.g. when compiling with:

```sh
rustc -Zlocation-detail=line
```

only the line number will be real. The file and column will always be a dummy value like

    thread 'main' panicked at 'error message', <redacted>:192:0


# Drawbacks
[drawbacks]: #drawbacks

## Code bloat

Previously, all calls to `unwrap()` and `expect()` referred to the same location. Therefore, the
panicking branch will only needed to reuse a pointer to a single global tuple.

After this RFC is implemented, the panicking branch will need to allocate space to store the varying caller location,
so the number of instructions per `unwrap()`/`expect()` will increase.

The optimizer will lose the opportunity to consolidate all jumps to the panicking branch. Before
this RFC, LLVM would optimize `a.unwrap() + b.unwrap()`, to something like

```rust
if (a.tag != SOME || b.tag != SOME) {
    panic(&("called `Option::unwrap()` on a `None` value", "src/libcore/option.rs", 335, 20));
}
a.value_of_some + b.value_of_some
```

After this RFC, LLVM can only lower this to

```rust
if (a.tag != SOME) {
    panic(&("called `Option::unwrap()` on a `None` value", "1.rs", 1, 1));
}
if (b.tag != SOME) {
    panic(&("called `Option::unwrap()` on a `None` value", "1.rs", 1, 14));
}
a.value_of_some + b.value_of_some
```

One can use `-Z location-detail` to get the old optimization behavior.

## Narrow solution scope

`#[track_caller]` is only useful in solving the "get caller location" problem. Introducing an
entirely new feature just for this problem seems wasteful.

[Default function arguments](#default-function-arguments) is another possible solution for this
problem but with much wider application.

## Confusing scoping rule

Consts, statics and closures are separate MIR items, meaning the following marked places will *not*
get caller locations:

```rust
#[track_caller]
fn foo() {
    static S: Location = Location::caller(); // will get actual location instead
    let f = || Location::caller();   // will get actual location instead
    Location::caller(); // this one will get caller location
}
```

This is confusing, but if we don't support this, we will need two `panic!` macros which is not a
better solution.

Clippy could provide a lint against using `Location::caller()` outside of `#[track_caller]`.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Rationale

This RFC tries to abide by the following restrictions:

1. **Precise caller location**. Standard library functions which commonly panic will report the
    source location as where the user called them. The source location should never point inside the
    standard library. Examples of these functions include `Option::unwrap` and `HashMap::index`.

2. **Source compatibility**. Users should never need to modify existing source code to benefit from
    the improved precision.

3. **Debug-info independence**. The precise caller location can still be reported even after
    stripping of debug information, which is very common on released software.

4. **Interface independence**. The implementation of a trait should be able to decide whether to
    accepts the caller information; it shouldn't require the trait itself to enforce it. It
    should not affect the signature of the function. This is an extension of rule 2, since the
    `Index` trait is involved in `HashMap::index`. The stability of `Index` must be upheld, e.g. it
    should remain object-safe, and existing implementions should not be forced to accept the caller
    location.

Restriction 4 "interface independence" is currently not implemented due to lack of
post-monomorphized MIR pass, but implementing `#[track_caller]` as a language feature follows this
restriction.

## Alternatives

### 🚲 Name of everything 🚲

* Is `#[track_caller]` an accurate description?
* Should we move `std::panic::Location` into `core`, or just use a 3-tuple to represent the
    location? Note that the former is advocated in [RFC 2070].
* Is `Location::caller()` properly named?

### Using an ABI instead of an attribute

```rust
pub extern "implicit-caller-location" fn my_unwrap() {
    panic!("oh no");
}
```

Compared with attributes, an ABI is a more natural way to tell the post-typechecking steps about
implicit parameters, pioneered by the `extern "rust-call"` ABI. However, creating a new ABI will
change the type of the function as well, causing the following statement to fail:

```rust
let f: fn(Option<u32>) -> u32 = Option::unwrap;
//~^ ERROR: [E0308]: mismatched types
```

Making this pass will require supporting implicitly coercing `extern "implicit-caller-location" fn`
pointer to a normal function pointer. Also, an ABI is not powerful enough to implicitly insert a
parameter, making it less competitive than just using an attribute.

### Repurposing `file!()`, `line!()`, `column!()`

We could change the meaning of `file!()`, `line!()` and `column!()` so they are only converted to
real constants after redirection (a MIR or trans pass) instead of early during macro expansion (an
AST pass). Inside `#[track_caller]` functions, these macros behave as this RFC's
`caller_location()`. The drawback is using these macro will have different values at compile time
(e.g. inside `include!(file!())`) vs. runtime.

### Inline MIR

Introduced as an [alternative to RFC 1669][inline_mir], instead of the `caller_location()` intrinsic,
we could provide a full-fledged inline MIR macro `mir!` similar to the inline assembler:

```rust
#[track_caller]
fn unwrap(self) -> T {
    let file: &'static str;
    let line: u32;
    let column: u32;
    unsafe {
        mir! {
            StorageLive(file);
            file = const $CallerFile;
            StorageLive(line);
            line = const $CallerLine;
            StorageLive(column);
            column = const $CallerColumn;
            goto -> 'c;
        }
    }
    'c: {
        panic!("{}:{}:{}: oh no", file, line, column);
    }
}
```

The problem of `mir!` in this context is trying to kill a fly with a sledgehammer. `mir!` is a very
generic mechanism which requires stabilizing the MIR syntax and considering the interaction with
the surrounding code. Besides, `#[track_caller]` itself still exists and the magic constants
`$CallerFile` etc are still magical.

### Default function arguments

Assume this is solved by implementing [RFC issue 323].

```rust
fn unwrap(file: &'static str = file!(), line: u32 = line!(), column: u32 = column!()) -> T {
    panic!("{}:{}:{}: oh no", file, line, column);
}
```

Default arguments was a serious contender to the better-caller-location problem as this is usually
how other languages solve it.

| Language | Syntax |
|:---------|:-------|
| [Swift] | `func unwrap(file: String = #file, line: Int = #line) -> T` |
| [D] | `T unwrap(string file = __FILE__, size_t line = __LINE__)` |
| [C#] 5+ | `T Unwrap([CallerFilePath] string file = "<n/a>", [CallerLineNumber] int line = 0)` |
| [Haskell] with GHC | `unwrap :: (?callstack :: CallStack) => Maybe t -> t` |
| [C++] with GCC 4.8+ | `T unwrap(const char* file = __builtin_FILE(), int line = __builtin_LINE())` |

A naive solution will violate restriction 4 "interface independence": adding the `file, line, column`
arguments to `index()` will change its signature. This can be resolved if this is taken into
account.

```rust
impl<'a, K, Q, V> Index<&'a Q> for BTreeMap<K, V>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    // This should satisfy the trait even if the trait specifies
    // `fn index(&self, idx: Idx) -> &Self::Output`
    #[inline]
    fn index(&self, key: &Q, file: &'static str = file!(), line: u32 = line!(), column: u32 = column!()) -> &V {
        self.get(key).expect("no entry found for key", file, line, column)
    }
}
```

This can be resolved if the future default argument proposal takes this into account. But again,
this feature itself is going to be large and controversial.

### Semantic inlining

Treat `#[track_caller]` as the same as a very forceful `#[inline(always)]`. This eliminates the
procedural macro pass. This was the approach suggested in the first edition of this RFC, since the
target functions (`unwrap`, `expect`, `index`) are just a few lines long. However, it experienced
push-back from the community as:

1. Inlining causes debugging to be difficult.
2. It does not work with recursive functions.
3. People do want to apply the attribute to long functions.
4. The expected usage of "semantic inlining" and traditional inlining differ a lot, continue calling
    it inlining may confuse beginners.

Therefore the RFC is changed to the current form, and the inlining pass is now described as just an
implementation detail.

### Design-by-contract

This is inspired when investigating the difference in
["my fault" vs "your fault"](#my-fault-vs-your-fault). We incorporate ideas from [design-by-contract]
(DbC) by specifying that "your fault" is a kind of contract violation. Preconditions are listed as
part of the function signature, e.g.

```rust
// declaration
extern {
    #[precondition(fd >= 0, "invalid file descriptor {}", fd)]
    fn close_fd(fd: c_int);
}

// declaration + defintion
#[precondition(option.is_some(), "Trying to unwrap None")]
fn unwrap<T>(option: Option<T>) -> T {
    match option {
        Some(t) => t,
        None => unsafe { std::mem::unchecked_unreachable() },
    }
}
```

Code that appears in the `#[precondition]` attribute should be copied to caller site, so when the
precondition is violated, they can get the caller's location.

Specialization should be treated like subtyping, where preconditions can be *weakened*:

```rust
trait Foo {
    #[precondition(condition_1)]
    fn foo();
}

impl<T: Debug> Foo for T {
    #[precondition(condition_2a)]
    #[precondition(condition_2b)]
    default fn foo() { ... }
}

impl Foo for u32 {
    #[precondition(condition_3)]
    fn foo() { ... }
}

assert!(condition_3 || (condition_2a && condition_2b) || condition_1);
// ^ automatically inserted when the following is called...
<u32 as Foo>::foo();
```

Before Rust 1.0, there was the [`hoare`] compiler plugin which introduces DbC using the similar
syntax. However, the conditions are expanded inside the function, so the assertions will not fail
with the caller's location. A proper solution will be similar to what this RFC proposes.

[design-by-contract]: https://en.wikipedia.org/wiki/Design_by_contract
[`hoare`]: https://crates.io/crates/hoare

## Non-viable alternatives

Many alternatives have been proposed before but failed to satisfy the restrictions laid out in the
[Rationale](#rationale) subsection, thus should *not* be considered viable alternatives within this
RFC, at least at the time being.

### Macros

The `unwrap!()` macro introduced in [RFC 1669] allows the user to write `unwrap!(x)` instead of
`x.unwrap()`.

A similar solution is introducing a `loc!()` macro that expands to
`concat!(file!(), ":", line!(), ":", column!())`, so user writes `x.expect(loc!())` instead of
`x.unwrap()`.

There is even the [`better_unwrap` crate](https://github.com/abonander/better_unwraps) that
automatically rewrites all `unwrap()` and `expect()` inside a module to provide the caller location
through a procedural attribute.

All of these are non-viable since they require the user to actively change their source code, thus
violating restriction 2 "source compatibility", ~~unless we are willing to drop the `!` from
macros~~.

All pre-typeck rewrites are prone to false-positive failures affecting unrelated types that have an
`unwrap()` method. Post-typeck rewrites are no different from this RFC.

### Backtrace

When given debug information (DWARF section/file on Linux, `*.pdb` file on Windows, `*.dSYM` folder
on macOS), the program is able to obtain the source code location for each address. This solution is
often used in runtime-heavy languages like Python, Java and [Go].

For Rust, however:

* The debug information is usually not provided in release mode.

    In particular, `cargo` defaults to disabling debug symbols in release mode (this default can
    certainly be changed). `rustc` itself is tested in CI and distributed in release mode, so
    getting a usable location in release mode is a real concern (see also [RFC 1417] for why it was
    disabled in the official distribution in the first place).

    Even if this is generated, the debug symbols are generally not distributed to end-users, which
    means the error reports will only contain numerical addresses. This can be seen as a benefit, as
    the implementation detail won't be exposed, but how to submit/analyze an error report would be
    out-of-scope for this RFC.

* There are multiple issues preventing us from relying on debug info nowadays.

    Issues [24346]  (*Backtrace does not include file and line number on non-Linux platforms*) and
    [42295]  (*Slow backtrace on panic*) and are still not entirely fixed. Even after the debuginfo
    is properly handled, if we decide not to expose the whole the full stacktrace, we may still need
    to reopen pull request [40264]  (*Ignore more frames on backtrace unwinding*).

    These signal that debuginfo support is not reliable enough if we want to solve the unwrap/expect
    issue now.

These drawbacks are the main reason why restriction 3 "debug-info independence" is added to the
motivation.

(A debuginfo-based stack trace proposal can be found at [RFC 2154].)

### `SourceContext` generic parameter

Introduced as an [alternative in RFC 1669][source_context], inspired by GHC's implicit parameter:

```rust
fn unwrap<C: SourceContext = CallerSourceContext>(self) -> T {
    panic!("{}: oh no", C::default());
}
```

The `CallerSourceContext` lang item will instruct the compiler to create a new type implementing
`SourceContext` whenever `unwrap()` is instantiated.

Unfortunately this violates restriction 4 "interface independence". This solution cannot apply to
`HashMap::index` as this will require a change of the method signature of `index()` which has been
stabilized. Methods applying this solution will also lose object-safety.

The same drawback exists if we base the solution on [RFC 2000]  (*const generics*).

# Unresolved questions
[unresolved]: #unresolved-questions

* If we want to support adding `#[track_caller]` to trait methods, the redirection
    pass/query/whatever should be placed after monomorphization, not before. Currently the RFC
    simply prohibit applying `#[track_caller]` to trait methods as a future-proofing measure.

* Diverging functions should be supported.

* The closure `foo::{{closure}}` should inherit most attributes applied to the function `foo`, in
    particular `#[inline]`, `#[cold]`, `#[naked]` and also the ABI. Currently a procedural macro
    won't see any of these, nor would there be anyway to apply these attributes to a closure.
    Therefore, `#[rustc_implicit_caller_location]` currently will reject `#[naked]` and ABI, and
    leaving `#[inline]` and `#[cold]` mean no-op. There is no semantic reason why these cannot be
    used though.

[RFC 1669]: https://github.com/rust-lang/rfcs/pull/1669
[24346]: https://github.com/rust-lang/rust/issues/24346
[42295]: https://github.com/rust-lang/rust/issues/42295
[issue 33880]: https://github.com/rust-lang/rust/issues/33880
[RFC issue 1744]: https://github.com/rust-lang/rfcs/issues/1744
[RFC issue 323]: https://github.com/rust-lang/rfcs/issues/323
[RFC 2070]: https://github.com/rust-lang/rfcs/pull/2070
[RFC 2000]: https://github.com/rust-lang/rfcs/pull/2000
[40264]: https://github.com/rust-lang/rust/issues/40264
[RFC 1417]: https://github.com/rust-lang/rfcs/issues/1417
[RFC 2154]: https://github.com/rust-lang/rfcs/pull/2154

[a]: https://internals.rust-lang.org/t/rfrfc-better-option-result-error-messages/2904
[b]: https://internals.rust-lang.org/t/line-info-for-unwrap-expect/3753
[c]: https://internals.rust-lang.org/t/better-panic-location-reporting-for-unwrap-and-friends/5042

[source_context]: https://github.com/rust-lang/rfcs/pull/1669#issuecomment-231896669
[inline_mir]: https://github.com/rust-lang/rfcs/pull/1669#issuecomment-231031865
[Swift]: https://developer.apple.com/swift/blog/?id=15
[D]: https://dlang.org/spec/traits.html#specialkeywords
[C#]: https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/caller-information
[Haskell]: https://ghc.haskell.org/trac/ghc/wiki/ExplicitCallStack/ImplicitLocations
[Go]: https://golang.org/pkg/runtime/#Caller
[C++]: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fLINE

[inlining]: https://en.wikipedia.org/wiki/Inline_expansion
