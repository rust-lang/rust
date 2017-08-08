- Feature Name: `implicit_caller_location`
- Start Date: 2017-07-31
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

----

# Summary
[summary]: #summary

Enable accurate caller location reporting during panic in `{Option, Result}::{unwrap, expect}` with
the following changes:

1. Support the `#[implicit_caller_location]` function attribute, which guarantees a function has
    access to the caller information.
2. Adds an intrinsic function `caller_location()` (safe wrapper: `Location::caller()`) to retrieve
    the caller's source location.

Example:

```rust
#![feature(implicit_caller_location)]
use std::panic::Location;

#[implicit_caller_location]
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
    - [Adding caller location implicitly](#adding-caller-location-implicitly)
    - [Location type](#location-type)
    - [Why do we use implicit caller location](#why-do-we-use-implicit-caller-location)
- [Reference-level explanation](#reference-level-explanation)
    - [Survey of panicking standard functions](#survey-of-panicking-standard-functions)
    - [Procedural attribute macro](#procedural-attribute-macro)
    - [Redirection (MIR inlining)](#redirection-mir-inlining)
    - [Standard libraries](#standard-libraries)
    - [Runtime-free backtrace for `?` operator](#runtime-free-backtrace-for--operator)
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
    - [Non-viable alternatives](#non-viable-alternatives)
        - [Macros](#macros)
        - [Backtrace](#backtrace)
        - [`SourceContext` generic parameter](#sourcecontext-generic-parameter)
- [Unresolved questions](#unresolved-questions)

<!-- /TOC -->

# Motivation
[motivation]: #motivation

It is well-known that the error message reported by `unwrap()` is useless.

```text
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', /checkout/src/libcore/option.rs:335
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```

There have been numerous discussions ([a], [b], [c]) that wants `unwrap()` and friends to provide
the better information to locate the panic. Previously, [RFC 1669] attempted to address this by
introducing the `unwrap!(x)` macro to the standard library, but it was closed since the `x.unwrap()`
convention is too entrenched.

This RFC tries to introduce line numbers into `unwrap()` without requiring users to adapt a new
idiom, i.e. the user should be able to see the precise location without changing any of the source
code.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Let's reimplement `unwrap()`

`unwrap()` and `expect()` are two methods on `Option` and `Result` that are commonly used when you
are *absolutely sure* they only contains a successful value and you want to extract it.

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

This trivial implementation, however, will only report the panic happens inside `my_unwrap`. This is
pretty useless, since it is the caller of `my_unwrap` that have made the wrong assumption!

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

The trivial solution would be requiring user to provide `file!()`, `line!()` and `column!()`. A
slightly more ergonomic solution would be changing `my_unwrap` to a macro, thus these constants can
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

But what if you have already published the `my_unwrap` crate that has thousands of users, and you
want to maintain API stability? Before Rust 1.XX, the builtin `unwrap()` has the same problem!

## Adding caller location implicitly

The reason a `my_unwrap!` macro works is because it copy-and-paste the entire content of its macro
definition everytime it is used.

```rust
println!("args[1] = {}", my_unwrap!(args().nth(1)));
println!("args[2] = {}", my_unwrap!(args().nth(2)));
...

// is equivalent to:

println!("args[1] = {}", my_unwrap(args().nth(1), file!(), line!(), column!()));
println!("args[1] = {}", my_unwrap(args().nth(2), file!(), line!(), column!()));
...
```

What if we can instruct the compiler to automatically fill in the file-line-column? Rust supported
this starting from 1.YY with the `#[implicit_caller_location]` attribute:

```rust
// 3.rs
#![feature(implicit_caller_location)]
use std::env::args;
#[implicit_caller_location]  // <-- Just add this!
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

Now we magically have truly reproduced how the built-in `unwrap()` is implemented.

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

What `#[implicit_caller_location]` does is an automated version of what you've seen last section.
The attribute will copy `my_unwrap` into a new function `my_unwrap_at_source_location`, which will
accept the caller's location as an additional argument. The attribute also instructs the compiler
that, "whenever you see someone calls `my_unwrap(x)`, replace it with
`my_unwrap_at_source_location(x, file!(), line!(), column!())` instead!". This allows us to maintain
the stability guarantee, while allowing user to get the new behavior with just one recompile.

## Location type

Let's enhance `my_unwrap`, say, we also log a message to the log file before panicking? We would
need to get the caller's location as a value. This is supported using the method
`Location::caller()`:

```rust
use std::panic::Location;
#[implicit_caller_location]
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

## Why do we use implicit caller location

If you are learning Rust alongside other languages, you may wonder why Rust obtains the caller
information in such a strange way. There are two restrictions that forces us to adapt this solution:

1. Programmatic access to the stack backtrace is often used in interpreted or runtime-heavy
    languages like Python and Java. However, the stack backtrace is not suitable as the only
    solution for systems languages like Rust, because optimization often collapses multiple levels
    of function calls, and in some embedded system the backtrace may even be unavailable.

2. Rust does not (yet) support default function arguments or function overloading, because it badly
    interferes with type inference. Therefore, solutions that uses default function arguments
    alongside normal arguments are also ruled out. Default-function-arguments-based solutions are
    often used in languages that does not perform inference higher than statement level, e.g. Swift
    and C#.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Survey of panicking standard functions

Many standard functions may panic. These are divided into three categories, on whether they should
receive caller information despite the inlining cost associated with it.

(The list of functions is not exhaustive. Only those with a "Panics" section in the documentation
are included.)

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
    rare. Often the panic information contains enough clue to fix the error without backtrace.
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

3. **Not needed.** Panics from these indicate silly programmer error and the panic itself have
    enough clue to let programmers figure out where did the error comes from.

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

This RFC only advocates adding the `#[implicit_caller_location]` attribute to the `unwrap` and
`expect` functions. The `index` and `index_mut` functions should also have it if possible, but
currently blocked by lack of post-monomorphized MIR pass.

## Procedural attribute macro

The `#[implicit_caller_location]` attribute will modify a function at AST level and MIR level,
without touching type-checking (HIR level) or the low-level LLVM passes.

It will first wrap the body of the function in a closure, and then call it:

```rust
#[implicit_caller_location]
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

The reason for this is to split the function into two: the function `foo` itself, and the closure
`foo::{{closure}}` in it. (Technically: it is the simplest way to create two `DefId` at HIR level as
far as I know.)

The function signature of `foo` remains unchanged, so typechecking can proceed normally. The
attribute will be replaced by `#[rustc_implicit_caller_location]` to let the compiler internals
continue to treat it specially. `#[inline]` is added so external crate can see through `foo` to find
`foo::{{closure}}`.

The closure `foo::{{closure}}` is a proper function, that the compiler can write calls directly to
`foo::{{closure}}`, skipping `foo`. Multiple calls to `foo` from different location can be done via
calling `foo::{{closure}}` directly, instead of copying the function body everytime which will bloat
the binary size.

The intrinsic `caller_location()` is a placeholder which will be replaced by the actual caller
location when one calls `foo::{{closure}}` directly.

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

Then, we will further replace the `caller_location()` intrinsic according to where `foo` is called.
If it is called from an ordinary function, it would be replaced by the callsite's location:

```rust
// for ordinary functions,

_c = call std::intrinsics::caller_location() -> 'bbt;

// will become:

_c = Location { file: file!(), line: line!(), column: column!() };
goto -> 'bbt;
```

On the other hand, if it is called from an `#[implicit_caller_location]`'s closure e.g.
`foo::{{closure}}`, the intrinsic will be replaced by the closure argument `__location` instead, so
the caller location can propagate directly

```rust
// for #[implicit_caller_location] closures,

_c = call std::intrinsics::caller_location() -> 'bbt;

// will become:

_c = __location;
goto -> 'bbt;
```

These steps are very similar to inlining, and thus the first proof-of-concept is implemented
directly as a variant of MIR inliner (but a separate pass). This also means the redirection pass
currently suffers from all disadvantages of the MIR inliner, namely:

* Locations will not be propagated into diverging functions (`fn() -> !`), since inlining them is
    not supported yet.

* MIR passes are run *before* monomorphization, meaning `#[implicit_caller_location]` currently
    **cannot** be used on trait items:

```rust
trait Trait {
    fn unwrap(&self);
}
impl Trait for u64 {
    #[implicit_caller_location] //~ ERROR: `#[implicit_caller_location]` is not supported for trait items yet.
    fn unwrap(&self) {}
}
```

To support trait items, the redirection pass must be run as post-monomorphized MIR pass (which does
not exist yet), or a custom LLVM inlining pass which can extract the caller's source location. This
prevents the `Index` trait from having `#[implicit_caller_location]` yet.

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

Currently the redirection pass always run before the inlining pass. If the redirection pass is run
after the normal MIR inlining pass, the normal MIR inliner must treat `#[implicit_caller_location]`
as `#[inline(never)]`.

The closure `foo::{{closure}}` must never be inlined before the redirection pass.

When `#[implicit_caller_location]` functions are called dynamically, no inlining will occur, and
thus it cannot take the location of the caller. Currently this will report where the function is
declared. Taking the address of such functions must be allowed due to backward compatibility. (If
post-monomorphized MIR pass exists, methods via trait objects would be another case of calling
`#[implicit_caller_location]` functions without caller location.)

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
re-exported in `libstd`.

Thanks to how `#[implicit_caller_location]` is implemented, we could provide a safe wrapper around
the `caller_location()` intrinsic:

```rust
impl<'a> Location<'a> {
    #[implicit_caller_location]
    pub fn caller() -> Location<'static> {
        unsafe {
            ::intrinsics::caller_location()
        }
    }
}
```

The `panic!` macro is modified to use `Location::caller()` (or the intrinsic directly) so it can
report the caller location inside `#[implicit_caller_location]`.

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
instead of tuples, so one should consider changing their signature. But this is out-of-scope for
this RFC.

`panic!` is often used about of `#[implicit_caller_location]` functions. In those cases, the
`caller_location()` intrinsic will pass unchanged through all MIR passes into trans. As a fallback,
the intrinsic will expand to `Location { file: file!(), line: line!(), col: column!() }` during
trans.

## Runtime-free backtrace for `?` operator

The standard `Try` implementations could participate in specialization, so external crates like
`error_chain` could provide a specialized impl that prepends the caller location into the callstack
everytime `?` is used.

```rust
// libcore:
impl<T, E> Try for Result<T, E> {
    type Ok = T;
    type Error = E;

    fn into_result(self) -> Self { self }
    fn from_ok(v: Self::Ok) -> Self { Ok(v) }
    default fn from_error(e: Self::Error) -> Self { Err(e) }
//  ^~~~~~~
}

// my_crate:
impl<T> Try for Result<T, my_crate::error::Error> {
    #[implicit_caller_location]
    fn from_error(mut e: Self::Error) -> Self {
        e.call_stack.push(Location::caller());
        Err(e)
    }
}
```

## Location detail control

An unstable flag `-Z location-detail` is added to `rustc` to control how much factural detail will
be emitted when using `caller_location()`. User can toggle `file`, `line` and `column` separately,
e.g. when compiling with:

```sh
rustc -Zlocation-detail=line
```

only the line number will be real, and the file and column will always be dummy value like

    thread 'main' panicked at 'error message', <redacted>:192:0


# Drawbacks
[drawbacks]: #drawbacks

## Code bloat

Previously, all call to `unwrap()` and `expect()` will refer to the same location. Therefore, the
panicking branch will only need to reuse a pointer to a single global tuple.

After this RFC is implemented, the panicking branch will need to allocate space to store the varying caller location,
so the number of instructions per `unwrap()`/`expect()` will increase.

Also the optimizer will lose the opportunity to consolidate all jumps to the panicking branch, e.g. with the code
`a.unwrap() + b.unwrap()`, before this RFC LLVM will optimize it to something like

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

(One could use `-Z location-detail` to get the old optimization behavior)

## Narrow solution scope

`#[implicit_caller_location]` is only useful in solving the "get caller location" problem.
Introducing an entirely new feature just for this problem seems wasteful.

[Default function arguments](#default-function-arguments) is another possible solution for this
problem but with much wider application.

## Confusing scoping rule

Consts, statics and closures are separate MIR items, meaning the following marked places will *not*
get caller locations:

```rust
#[implicit_caller_location]
fn foo() {
    static S: Location = Location::caller(); // will get actual location instead
    let f = || Location::caller();   // will get actual location instead
    Location::caller(); // this one will get caller location
}
```

This is confusing, but if we don't support this, we will need two `panic!` macros which is not much
better.

Clippy could provide a lint against using `Location::caller()` outside of
`#[implicit_caller_location]`.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Rationale

This RFC tries to abide by the following restrictions:

1. **Precise caller location**. Standard library functions which commonly panics will report the
    source location at where the user call them. The source location should never point inside the
    standard library. Examples of these functions include `Option::unwrap` and `HashMap::index`.

2. **Source compatibility**. User should never need to modify existing source code to benefit from
    the improved precision.

3. **Debug-info independence**. The precise caller location can still be reported even after
    stripping of debug information, which is very common on released software.

4. **Interface independence**. The implementation of a trait should be able to decide whether to
    accept a the caller information. It shouldn't require the trait itself to enforce it, i.e. it
    should not affect the signature of the function. This is an extension of rule 2, since the
    `Index` trait is involved in `HashMap::index`. The stability of `Index` must be upheld, e.g. it
    should remain object-safe, and existing implementions should not be forced to accept the caller
    location.

Restriction 4 "interface independence" is currently not implemented due to lack of
post-monomorphized MIR pass, but `#[implicit_caller_location]` itself as a language feature follows
this restriction.

## Alternatives

### 🚲 Name of everything 🚲

* Is `#[implicit_caller_location]` an accurate description?
* Should we move `std::panic::Location` into `core`, or just use a 3-tuple to represent the
    location? Note that the former is advocated in [RFC 2070].
* Should caller location propagation be implicit (currently) or explicit?

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
AST pass). Inside `#[implicit_caller_location]` functions, these macros behave as this RFC's
`caller_location()`. The drawback is using these macro will have different values in compile time
(e.g. inside `include!(file!())`) vs. runtime.

### Inline MIR

Introduced as [alternative to RFC 1669][inline_mir], instead of the `caller_location()` intrinsic,
we could provide a full-fledged inline MIR macro `mir!` similar to the inline assembler:

```rust
#[implicit_caller_location]
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
generic mechanism, which requiring stabilizing the MIR syntax and considering the interaction with
the surrounding code. Besides, `#[implicit_caller_location]` itself still exists and the magic
constants `$CallerFile` etc are still magic.

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

Treat `#[implicit_caller_location]` as the same as a very forceful `#[inline(always)]`. This
eliminates the procedural macro pass. However, typical implementation will still need to manually
implement the two functions, otherwise the code will become very bloated after all the inlining.

## Non-viable alternatives

Many alternatives have been proposed before but failed to satisfy the restrictions laid out in the
[Rationale](#rationale) subsection, thus should *not* be considered viable alternatives within this
RFC.

### Macros

The `unwrap!()` macro introduced in [RFC 1669] allows the user to write `unwrap!(x)` instead of
`x.unwrap()`.

Similar solution is introducing a `loc!()` macro that expands to
`concat!(file!(), ":", line!(), ":", column!())`, so user writes `x.expect(loc!())` instead of
`x.unwrap()`.

There is even the [`better_unwrap` crate](https://github.com/abonander/better_unwraps) that
automatically rewrites all `unwrap()` and `expect()` inside a module to provide the caller location
through a procedural attribute.

All of these are non-viable, since they require the user to actively change their source code, thus
violating restriction 2 "source compatibility", ~~unless we are willing to drop the `!` from
macros~~.

(Also, all pre-typeck rewrites are prone to false-positive affecting unrelated types that have an
`unwrap()` method. Post-typeck rewrites are no different from this RFC.)

### Backtrace

When given debug information (DWARF section on Linux, `*.pdb` file on Windows, `*.dSYM` folder on
macOS), the program is able to obtain the source code location for each address. This solution is
often used in runtime-heavy languages like Python, Java and [Go].

For Rust, however:

* The debug information is usually not provided in release mode.
* Normal inlining will make the caller information inaccurate.
* Issues [24346]  (*Backtrace does not include file and line number on non-Linux platforms*) and
    [42295]  (*Slow backtrace on panic*) and are still not entirely fixed.
* Backtrace may be disabled in resource-constrained environment.

These drawbacks are the main reason why restriction 3 "debug-info independence" is added to the
motivation.

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

# Unresolved questions
[unresolved]: #unresolved-questions

* Redirection pass should run after monomorphization, not before.

* Diverging functions should be supported.

* The closure `foo::{{closure}}` should inherit most attributes applied to the function `foo`, in
    particular `#[inline]`, `#[cold]` and `#[naked]`. Currently a procedural macro won't see any of
    these, nor would there be anyway to apply these attributes to a closure.

[RFC 1669]: https://github.com/rust-lang/rfcs/pull/1669
[24346]: https://github.com/rust-lang/rust/issues/24346
[42295]: https://github.com/rust-lang/rust/issues/42295
[issue 33880]: https://github.com/rust-lang/rust/issues/33880
[RFC issue 1744]: https://github.com/rust-lang/rfcs/issues/1744
[RFC issue 323]: https://github.com/rust-lang/rfcs/issues/323
[RFC 2070]: https://github.com/rust-lang/rfcs/pull/2070

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