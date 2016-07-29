- Feature Name: bang_type
- Start Date: 2015-07-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/1216
- Rust Issue: https://github.com/rust-lang/rust/issues/35121

# Summary

Promote `!` to be a full-fledged type equivalent to an `enum` with no variants.

# Motivation

To understand the motivation for this it's necessary to understand the concept
of empty types. An empty type is a type with no inhabitants, ie. a type for
which there is nothing of that type. For example consider the type `enum Never
{}`. This type has no constructors and therefore can never be instantiated. It
is empty, in the sense that there are no values of type `Never`. Note that
`Never` is not equivalent to `()` or `struct Foo {}` each of which have exactly
one inhabitant. Empty types have some interesting properties that may be
unfamiliar to programmers who have not encountered them before.

  * They never exist at runtime.
    Because there is no way to create one.

  * They have no logical machine-level representation.
    One way to think about this is to consider the number of bits required to
    store a value of a given type. A value of type `bool` can be in two
    possible states (`true` and `false`). Therefore to specify which state a
    `bool` is in we need `log2(2) ==> 1` bit of information. A value of type
    `()` can only be in one possible state (`()`). Therefore to specify which
    state a `()` is in we need `log2(1) ==> 0` bits of information. A value of
    type `Never` has no possible states it can be in. Therefore to ask which of
    these states it is in is a meaningless question and we have `log2(0) ==>
    undefined` (or `-∞`). Having no representation is not problematic as safe
    code never has reason nor ability to handle data of an empty type (as such
    data can never exist). In practice, Rust currently treats empty types as
    having size 0.

  * Code that handles them never executes.
    Because there is no value that it could execute with. Therefore, having a
    `Never` in scope is a static guarantee that a piece of code will never be
    run.

  * They represent the return type of functions that don't return.
    For a function that never returns, such as `exit`, the set of all values it
    may return is the empty set. That is to say, the type of all values it may
    return is the type of no inhabitants, ie. `Never` or anything isomorphic to
    it. Similarly, they are the logical type for expressions that never return
    to their caller such as `break`, `continue` and `return`.

  * They can be converted to any other type.
    To specify a function `A -> B` we need to specify a return value in `B` for
    every possible argument in `A`. For example, an expression that converts
    `bool -> T` needs to specify a return value for both possible arguments
    `true` and `false`:

    ```rust
    let foo: &'static str = match x {
      true  => "some_value",
      false => "some_other_value",
    };
    ```

    Likewise, an expression to convert `() -> T` needs to specify one value,
    the value corresponding to `()`:

    ```rust
    let foo: &'static str = match x {
      ()  => "some_value",
    };
    ```

    And following this pattern, to convert `Never -> T` we need to specify a
    `T` for every possible `Never`. Of which there are none:

    ```rust
    let foo: &'static str = match x {
    };
    ```

    Reading this, it may be tempting to ask the question "what is the value of
    `foo` then?". Remember that this depends on the value of `x`. As there are
    no possible values of `x` it's a meaningless question and besides, the
    fact that `x` has type `Never` gives us a static guarantee that the match
    block will never be executed.

Here's some example code that uses `Never`. This is legal rust code that you
can run today.

```rust
use std::process::exit;

// Our empty type
enum Never {}

// A diverging function with an ordinary return type
fn wrap_exit() -> Never {
    exit(0);
}

// we can use a `Never` value to diverge without using unsafe code or calling
// any diverging intrinsics
fn diverge_from_never(n: Never) -> ! {
    match n {
    }
}

fn main() {
    let x: Never = wrap_exit();
    // `x` is in scope, everything below here is dead code.

    let y: String = match x {
        // no match cases as `Never` has no variants
    };

    // we can still use `y` though
    println!("Our string is: {}", y);

    // we can use `x` to diverge
    diverge_from_never(x)
}
```

This RFC proposes that we allow `!` to be used directly, as a type, rather than
using `Never` (or equivalent) in its place. Under this RFC, the above code
could more simply be written.

```rust
use std::process::exit;

fn main() {
    let x: ! = exit(0);
    // `x` is in scope, everything below here is dead code.

    let y: String = match x {
        // no match cases as `Never` has no variants
    };

    // we can still use `y` though
    println!("Our string is: {}", y);

    // we can use `x` to diverge
    x
}
```

So why do this? AFAICS there are 3 main reasons

  * **It removes one superfluous concept from the language and allows diverging
    functions to be used in generic code.**

    Currently, Rust's functions can be divided into two kinds: those that
    return a regular type and those that use the `-> !` syntax to mark
    themselves as diverging. This division is unnecessary and means that
    functions of the latter kind don't play well with generic code.

    For example: you want to use a diverging function where something expects a
    `Fn() -> T`

    ```rust
    fn foo() -> !;
    fn call_a_fn<T, F: Fn() -> T>(f: F) -> T;

    call_a_fn(foo) // ERROR!
    ```

    Or maybe you want to use a diverging function to implement a trait method
    that returns an associated type:

    ```rust
    trait Zog {
        type Output
        fn zog() -> Output;
    };

    impl Zog for T {
        type Output = !;                    // ERROR!
        fn zog() -> ! { panic!("aaah!") };  // ERROR!
    }
    ```

    The workaround in these cases is to define a type like `Never` and use it
    in place of `!`. You can then define functions `wrap_foo` and `unwrap_zog`
    similar to the functions `wrap_exit` and `diverge_from_never` defined
    earlier. It would be nice if this workaround wasn't necessary.

  * **It creates a standard empty type for use throughout rust code.**

    Empty types are useful for more than just marking functions as diverging.
    When used in an enum variant they prevent the variant from ever being
    instantiated. One major use case for this is if a method needs to return a
    `Result<T, E>` to satisfy a trait but we know that the method will always
    succeed.

    For example, here's a saner implementation of `FromStr` for `String` than
    currently exists in `libstd`.

    ```rust
    impl FromStr for String {
        type Err = !;
        
        fn from_str(s: &str) -> Result<String, !> {
            Ok(String::from(s))
        }
    }
    ```

    This result can then be safely unwrapped to a `String` without using
    code-smelly things like `unreachable!()` which often mask bugs in code.

    ```rust
    let r: Result<String, !> = FromStr::from_str("hello");
    let s = match r {
        Ok(s)   => s,
        Err(e)  => match e {},
    }
    ```

    Empty types can also be used when someone needs a dummy type to implement a
    trait. Because `!` can be converted to any other type it has a trivial
    implementation of any trait whose only associated items are non-static
    methods. The impl simply matches on self for every method.

    Example:

    ```rust
    trait ToSocketAddr {
        fn to_socket_addr(&self) -> IoResult<SocketAddr>;
        fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>>;
    }

    impl ToSocketAddr for ! {
        fn to_socket_addr(&self) -> IoResult<SocketAddr> {
            match self {}
        }

        fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> {
            match self {}
        }
    }
    ```

    All possible implementations of this trait for `!` are equivalent. This is
    because any two functions that take a `!` argument and return the same type
    are equivalent - they return the same result for the same arguments and
    have the same effects (because they are uncallable).

    Suppose someone wants to call `fn foo<T: SomeTrait>(arg: Option<T>)` with
    `None`. They need to choose a type for `T` so they can pass `None::<T>` as
    the argument. However there may be no sensible default type to use for `T`
    or, worse, they may not have any types at their disposal that implement
    `SomeTrait`. As the user in this case is only using `None`, a sensible
    choice for `T` would be a type such that `Option<T>` can ony be `None`, ie.
    it would be nice to use `!`. If `!` has a trivial implementation of
    `SomeTrait` then the choice of `T` is truly irrelevant as this means `foo`
    doesn't use any associated types/lifetimes/constants or static methods of
    `T` and is therefore unable to distinguish `None::<A>` from `None::<B>`.
    With this RFC, the user could `impl SomeTrait for !` (if `SomeTrait`'s
    author hasn't done so already) and call `foo(None::<!>)`.

    Currently, `Never` can be used for all the above purposes. It's useful
    enough that @reem has written a package for it
    [here](https://github.com/reem/rust-void) where it is named `Void`. I've also
    invented it independently for my own projects and probably other people
    have as well. However `!` can be extended logically to cover all the above
    use cases. Doing so would standardise the concept and prevent different
    people reimplementing it under different names.

  * **Better dead code detection**

    Consider the following code:

    ```
    let t = std::thread::spawn(|| panic!("nope"));
    t.join().unwrap();
    println!("hello");

    ```
    Under this RFC: the closure body gets typed `!` instead of `()`, the `unwrap()`
    gets typed `!`, and the `println!` will raise a dead code warning. There's no
    way current rust can detect cases like that.

  * **Because it's the correct thing to do.**

    The empty type is such a fundamental concept that - given that it already
    exists in the form of empty enums - it warrants having a canonical form of
    it built-into the language. For example, `return` and `break` expressions
    should logically be typed `!` but currently seem to be typed `()`. (There
    is some code in the compiler that assigns type `()` to diverging
    expressions because it doesn't have a sensible type to assign to them).
    This means we can write stuff like this:

    ```rust
    match break {
      ()  => ...  // huh? Where did that `()` come from?
    }
    ```

    But not this:

    ```rust
    match break {} // whaddaya mean non-exhaustive patterns?
    ```

    This is just weird and should be fixed.

I suspect the reason that `!` isn't already treated as a canonical empty type
is just most people's unfamilarity with empty types. To draw a parallel in
history: in C `void` is in essence a type like any other. However it can't be
used in all the normal positions where a type can be used. This breaks generic
code (eg. `T foo(); T val = foo()` where `T == void`) and forces one to use
workarounds such as defining `struct Void {}` and wrapping `void`-returning
functions.

In the early days of programming having a type that contained no data probably
seemed pointless. After all, there's no point in having a `void` typed function
argument or a vector of `void`s. So `void` was treated as merely a special
syntax for denoting a function as returning no value resulting in a language
that was more broken and complicated than it needed to be.

Fifty years later, Rust, building on decades of experience, decides to fix C's
shortsightedness and bring `void` into the type system in the form of the empty
tuple `()`. Rust also introduces coproduct types (in the form of enums),
allowing programmers to work with uninhabited types (such as `Never`). However
rust also introduces a special syntax for denoting a function as never
returning: `fn() -> !`. Here, `!` is in essence a type like any other. However
it can't be used in all the normal positions where a type can be used. This
breaks generic code (eg. `fn() -> T; let val: T = foo()` where `T == !`) and
forces one to use workarounds such as defining `enum Never {}` and wrapping
`!`-returning functions.

To be clear, `!` has a meaning in any situation that any other type does. A `!`
function argument makes a function uncallable, a `Vec<!>` is a vector that can
never contain an element, a `!` enum variant makes the variant guaranteed never
to occur and so forth. It might seem pointless to use a `!` function argument
or a `Vec<!>` (just as it would be pointless to use a `()` function argument or
a `Vec<()>`), but that's no reason to disallow it. And generic code sometimes
requires it.

Rust already has empty types in the form of empty enums. Any code that could be
written with this RFC's `!` can already be written by swapping out `!` with
`Never` (sans implicit casts, see below). So if this RFC could create any
issues for the language (such as making it unsound or complicating the
compiler) then these issues would already exist for `Never`.

It's also worth noting that the `!` proposed here is *not* the bottom type that
used to exist in Rust in the very early days. Making `!` a subtype of all types
would greatly complicate things as it would require, for example, `Vec<!>` be a
subtype of `Vec<T>`. This `!` is simply an empty type (albeit one that can be
cast to any other type)

# Detailed design

Add a type `!` to Rust. `!` behaves like an empty enum except that it can be
implicitly cast to any other type. ie. the following code is acceptable:

```rust
let r: Result<i32, !> = Ok(23);
let i = match r {
    Ok(i)   => i,
    Err(e)  => e, // e is cast to i32
}
```

Implicit casting is necessary for backwards-compatibility so that code like the
following will continue to compile:

```rust
let i: i32 = match some_bool {
    true  => 23,
    false => panic!("aaah!"), // an expression of type `!`, gets cast to `i32`
}

match break {
    ()  => 23,  // matching with a `()` forces the match argument to be cast to type `()`
}
```
These casts can be implemented by having the compiler assign a fresh, diverging
type variable to any expression of type `!`.

In the compiler, remove the distinction between diverging and converging
functions. Use the type system to do things like reachability analysis.

Allow expressions of type `!` to be explicitly cast to any other type (eg.
`let x: u32 = break as u32;`)

Add an implementation for `!` of any trait that it can trivially implement. Add
methods to `Result<T, !>` and `Result<!, E>` for safely extracting the inner
value. Name these methods along the lines of `unwrap_nopanic`, `safe_unwrap` or
something.

# Drawbacks

Someone would have to implement this.

# Alternatives

  * Don't do this.
  * Move @reem's `Void` type into `libcore`. This would create a standard empty
    type and make it available for use in the standard libraries. If we were to
    do this it might be an idea to rename `Void` to something else (`Never`,
    `Empty` and `Mu` have all been suggested). Although `Void` has some
    precedence in languages like Haskell and Idris the name is likely to trip
    up people coming from a C/Java et al. background as `Void` is *not* `void`
    but it can be easy to confuse the two.

# Unresolved questions

`!` has a unique impl of any trait whose only items are non-static methods. It
would be nice if there was a way a to automate the creation of these impls.
Should `!` automatically satisfy any such trait? This RFC is not blocked on
resolving this question if we are willing to accept backward-incompatibilities
in questionably-valid code which tries to call trait methods on diverging
expressions and relies on the trait being implemented for `()`. As such, the
issue has been given [it's own RFC](https://github.com/rust-lang/rfcs/pull/1637).

