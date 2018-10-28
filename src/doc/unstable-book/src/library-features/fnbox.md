# `fnbox`

The tracking issue for this feature is [#28796]

[#28796]: https://github.com/rust-lang/rust/issues/28796

------------------------

As an analogy to `&dyn Fn()` and `&mut dyn FnMut()`, you may have expected
`Box<dyn FnOnce()>` to work. But it hadn't until the recent improvement!
`FnBox` had been a **temporary** solution for this until we are able to pass
trait objects by value.

See [`boxed_closure_impls`][boxed_closure_impls] for the newer approach.

[boxed_closure_impls]: library-features/boxed-closure-impls.html

## Usage

If you want to box `FnOnce` closures, you can use `Box<dyn FnBox()>` instead of `Box<dyn FnOnce()>`.

```rust
#![feature(fnbox)]

use std::boxed::FnBox;

fn main() {
    let resource = "hello".to_owned();
    // Create a boxed once-callable closure
    let f: Box<dyn FnBox() -> String> = Box::new(|| resource);

    // Call it
    let s = f();
    println!("{}", s);
}
```

## How `Box<dyn FnOnce()>` did not work

**Spoiler**: [`boxed_closure_impls`][boxed_closure_impls] actually implements
`Box<dyn FnOnce()>`! This didn't work because we lacked features like
[`unsized_locals`][unsized_locals] for a long time. Therefore, this section
just explains historical reasons for `FnBox`.

[unsized_locals]: language-features/unsized-locals.html

### First approach: just provide `Box` adapter impl

The first (and natural) attempt for `Box<dyn FnOnce()>` would look like:

```rust,ignore
impl<A, F: FnOnce<A> + ?Sized> FnOnce<A> for Box<F> {
    type Output = <F as FnOnce<A>>::Output;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        <F as FnOnce<A>>::call_once(*self, args)
    }
}
```

However, this doesn't work. We have to relax the `Sized` bound for `F` because
we expect trait objects here, but `*self` must be `Sized` because it is passed
as a function argument.

### The second attempt: add `FnOnce::call_box`

One may come up with this workaround: modify `FnOnce`'s definition like this:

```rust,ignore
pub trait FnOnce<Args> {
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
    // Add this new method
    extern "rust-call" fn call_box(self: Box<Self>, args: Args) -> Self::Output;
}
```

...and then, modify the `impl` like this:

```rust,ignore
impl<A, F: FnOnce<A> + ?Sized> FnOnce<A> for Box<F> {
    type Output = <F as FnOnce<A>>::Output;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        // We can use `call_box` here!
        <F as FnOnce<A>>::call_box(self, args)
    }
    // We'll have to define this in every impl of `FnOnce`.
    extern "rust-call" fn call_box(self: Box<Self>, args: A) -> Self::Output {
        <F as FnOnce<A>>::call_box(*self, args)
    }
}
```

What's wrong with this? The problem here is crates:

- `FnOnce` is in `libcore`, as it shouldn't depend on allocations.
- `Box` is in `liballoc`, as it:s the very allocated pointer.

It is impossible to add `FnOnce::call_box` because it is reverse-dependency.

There's another problem: `call_box` can't have defaults.
`default impl` from the specialization RFC may resolve this problem.

### The third attempt: add `FnBox` that contains `call_box`

`call_box` can't reside in `FnOnce`, but how about defining a new trait in
`liballoc`?

`FnBox` is almost a copy of `FnOnce`, but with `call_box`:

```rust,ignore
pub trait FnBox<Args> {
    type Output;

    extern "rust-call" fn call_box(self: Box<Self>, args: Args) -> Self::Output;
}
```

For `Sized` types (from which we coerce into `dyn FnBox`), we define
the blanket impl that proxies calls to `FnOnce`:

```rust,ignore
impl<A, F: FnOnce<A>> FnBox<A> for F {
    type Output = <F as FnOnce<A>>::Output;

    extern "rust-call" fn call_box(self: Box<Self>, args: A) -> Self::Output {
        // Here we assume `F` to be sized.
        <F as FnOnce<A>>::call_once(*self, args)
    }
}
```

Now it looks like that we can define `FnOnce` for `Box<F>`.

```rust,ignore
impl<A, F: FnBox<A> + ?Sized> FnOnce<A> for Box<F> {
    type Output = <F as FnOnce<A>>::Output;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        <F as FnBox<A>>::call_box(self, args)
    }
}
```

## Limitations of `FnBox`

### Interaction with HRTB

Firstly, the actual implementation is different from the one presented above.
Instead of implementing `FnOnce` for `Box<impl FnBox>`, `liballoc` only
implements `FnOnce` for `Box<dyn FnBox>`.

```rust,ignore
impl<'a, A, R> FnOnce<A> for Box<dyn FnBox<A, Output = R> + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        FnBox::call_box(*self, args)
    }
}

// Sendable variant
impl<'a, A, R> FnOnce<A> for Box<dyn FnBox<A, Output = R> + Send + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        FnBox::call_box(*self, args)
    }
}
```

The consequence is that the following example doesn't work:

```rust,compile_fail
#![feature(fnbox)]

use std::boxed::FnBox;

fn main() {
    let f: Box<dyn FnBox(&i32)> = Box::new(|x| println!("{}", x));
    f(42);
}
```

Note that `dyn FnBox(&i32)` desugars to
`dyn for<'r> FnBox<(&'r i32,), Output = ()>`.
It isn't covered in `dyn FnBox<A, Output = R> + 'a` or
`dyn FnBox<A, Output = R> + Send + 'a` due to HRTB.

### Interaction with `Fn`/`FnMut`

It would be natural to have the following impls:

```rust,ignore
impl<A, F: FnMut<A> + ?Sized> FnMut<A> for Box<F> {
    // ...
}
impl<A, F: Fn<A> + ?Sized> Fn<A> for Box<F> {
    // ...
}
```

However, we hadn't been able to write these in presense of `FnBox`
(until [`boxed_closure_impls`][boxed_closure_impls] lands).

To have `FnMut<A>` for `Box<F>`, we should have (at least) this impl:

```rust,ignore
// Note here we only impose `F: FnMut<A>`.
// If we can write `F: FnOnce<A>` here, that will resolve all problems.
impl<A, F: FnMut<A> + ?Sized> FnOnce<A> for Box<F> {
    // ...
}
```

Unfortunately, the compiler complains that it **overlaps** with our
`dyn FnBox()` impls. At first glance, the overlap must not happen.
The `A` generic parameter does the trick here: due to coherence rules,
a downstream crate may define the following impl:

```rust,ignore
struct MyStruct;
impl<'a> FnMut<MyStruct> for dyn FnBox<MyStruct, Output = ()> + 'a {
    // ...
}
```

The trait solver doesn't know that `A` is always a tuple type, so this is
still possible. With this in mind, the compiler emits the overlap error.

## Modification

For compatibility with [`boxed_closure_impls`][boxed_closure_impls],
we now have a slightly modified version of `FnBox`:

```rust,ignore
// It's now a subtrait of `FnOnce`
pub trait FnBox<Args>: FnOnce<Args> {
    // now uses FnOnce::Output
    // type Output;

    extern "rust-call" fn call_box(self: Box<Self>, args: Args) -> Self::Output;
}
```

## The future of `fnbox`

`FnBox` has long been considered a temporary solution for `Box<FnOnce>`
problem. Since we have [`boxed_closure_impls`][boxed_closure_impls] now,
it may be deprecated and removed in the future.
