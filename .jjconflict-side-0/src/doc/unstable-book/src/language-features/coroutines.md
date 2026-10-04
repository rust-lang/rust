# `coroutines`

The tracking issue for this feature is: [#43122]

[#43122]: https://github.com/rust-lang/rust/issues/43122

------------------------

The `coroutines` feature gate in Rust allows you to define coroutine or
coroutine literals. A coroutine is a "resumable function" that syntactically
resembles a closure but compiles to much different semantics in the compiler
itself. The primary feature of a coroutine is that it can be suspended during
execution to be resumed at a later date. Coroutines use the `yield` keyword to
"return", and then the caller can `resume` a coroutine to resume execution just
after the `yield` keyword.

Coroutines are an extra-unstable feature in the compiler right now. Added in
[RFC 2033] they're mostly intended right now as a information/constraint
gathering phase. The intent is that experimentation can happen on the nightly
compiler before actual stabilization. A further RFC will be required to
stabilize coroutines and will likely contain at least a few small
tweaks to the overall design.

[RFC 2033]: https://github.com/rust-lang/rfcs/pull/2033

A syntactical example of a coroutine is:

```rust
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut coroutine = #[coroutine] || {
        yield 1;
        return "foo"
    };

    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Complete("foo") => {}
        _ => panic!("unexpected value from resume"),
    }
}
```

Coroutines are closure-like literals which are annotated with `#[coroutine]`
and can contain a `yield` statement. The
`yield` statement takes an optional expression of a value to yield out of the
coroutine. All coroutine literals implement the `Coroutine` trait in the
`std::ops` module. The `Coroutine` trait has one main method, `resume`, which
resumes execution of the coroutine at the previous suspension point.

An example of the control flow of coroutines is that the following example
prints all numbers in order:

```rust
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut coroutine = #[coroutine] || {
        println!("2");
        yield;
        println!("4");
    };

    println!("1");
    Pin::new(&mut coroutine).resume(());
    println!("3");
    Pin::new(&mut coroutine).resume(());
    println!("5");
}
```

At this time the main use case of coroutines is an implementation
primitive for `async`/`await` and `gen` syntax, but coroutines
will likely be extended to other primitives in the future.
Feedback on the design and usage is always appreciated!

### The `Coroutine` trait

The `Coroutine` trait in `std::ops` currently looks like:

```rust
# #![feature(arbitrary_self_types, coroutine_trait)]
# use std::ops::CoroutineState;
# use std::pin::Pin;

pub trait Coroutine<R = ()> {
    type Yield;
    type Return;
    fn resume(self: Pin<&mut Self>, resume: R) -> CoroutineState<Self::Yield, Self::Return>;
}
```

The `Coroutine::Yield` type is the type of values that can be yielded with the
`yield` statement. The `Coroutine::Return` type is the returned type of the
coroutine. This is typically the last expression in a coroutine's definition or
any value passed to `return` in a coroutine. The `resume` function is the entry
point for executing the `Coroutine` itself.

The return value of `resume`, `CoroutineState`, looks like:

```rust
pub enum CoroutineState<Y, R> {
    Yielded(Y),
    Complete(R),
}
```

The `Yielded` variant indicates that the coroutine can later be resumed. This
corresponds to a `yield` point in a coroutine. The `Complete` variant indicates
that the coroutine is complete and cannot be resumed again. Calling `resume`
after a coroutine has returned `Complete` will likely result in a panic of the
program.

### Closure-like semantics

The closure-like syntax for coroutines alludes to the fact that they also have
closure-like semantics. Namely:

* When created, a coroutine executes no code. A closure literal does not
  actually execute any of the closure's code on construction, and similarly a
  coroutine literal does not execute any code inside the coroutine when
  constructed.

* Coroutines can capture outer variables by reference or by move, and this can
  be tweaked with the `move` keyword at the beginning of the closure. Like
  closures all coroutines will have an implicit environment which is inferred by
  the compiler. Outer variables can be moved into a coroutine for use as the
  coroutine progresses.

* Coroutine literals produce a value with a unique type which implements the
  `std::ops::Coroutine` trait. This allows actual execution of the coroutine
  through the `Coroutine::resume` method as well as also naming it in return
  types and such.

* Traits like `Send` and `Sync` are automatically implemented for a `Coroutine`
  depending on the captured variables of the environment. Unlike closures,
  coroutines also depend on variables live across suspension points. This means
  that although the ambient environment may be `Send` or `Sync`, the coroutine
  itself may not be due to internal variables live across `yield` points being
  not-`Send` or not-`Sync`. Note that coroutines do
  not implement traits like `Copy` or `Clone` automatically.

* Whenever a coroutine is dropped it will drop all captured environment
  variables.

### Coroutines as state machines

In the compiler, coroutines are currently compiled as state machines. Each
`yield` expression will correspond to a different state that stores all live
variables over that suspension point. Resumption of a coroutine will dispatch on
the current state and then execute internally until a `yield` is reached, at
which point all state is saved off in the coroutine and a value is returned.

Let's take a look at an example to see what's going on here:

```rust
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let ret = "foo";
    let mut coroutine = #[coroutine] move || {
        yield 1;
        return ret
    };

    Pin::new(&mut coroutine).resume(());
    Pin::new(&mut coroutine).resume(());
}
```

This coroutine literal will compile down to something similar to:

```rust
#![feature(arbitrary_self_types, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let ret = "foo";
    let mut coroutine = {
        enum __Coroutine {
            Start(&'static str),
            Yield1(&'static str),
            Done,
        }

        impl Coroutine for __Coroutine {
            type Yield = i32;
            type Return = &'static str;

            fn resume(mut self: Pin<&mut Self>, resume: ()) -> CoroutineState<i32, &'static str> {
                use std::mem;
                match mem::replace(&mut *self, __Coroutine::Done) {
                    __Coroutine::Start(s) => {
                        *self = __Coroutine::Yield1(s);
                        CoroutineState::Yielded(1)
                    }

                    __Coroutine::Yield1(s) => {
                        *self = __Coroutine::Done;
                        CoroutineState::Complete(s)
                    }

                    __Coroutine::Done => {
                        panic!("coroutine resumed after completion")
                    }
                }
            }
        }

        __Coroutine::Start(ret)
    };

    Pin::new(&mut coroutine).resume(());
    Pin::new(&mut coroutine).resume(());
}
```

Notably here we can see that the compiler is generating a fresh type,
`__Coroutine` in this case. This type has a number of states (represented here
as an `enum`) corresponding to each of the conceptual states of the coroutine.
At the beginning we're closing over our outer variable `foo` and then that
variable is also live over the `yield` point, so it's stored in both states.

When the coroutine starts it'll immediately yield 1, but it saves off its state
just before it does so indicating that it has reached the yield point. Upon
resuming again we'll execute the `return ret` which returns the `Complete`
state.

Here we can also note that the `Done` state, if resumed, panics immediately as
it's invalid to resume a completed coroutine. It's also worth noting that this
is just a rough desugaring, not a normative specification for what the compiler
does.
