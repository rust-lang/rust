# `boxed_closure_impls`

The tracking issue for this feature is [#48055]

[#48055]: https://github.com/rust-lang/rust/issues/48055

------------------------

This includes the following blanket impls for closure traits:

```rust,ignore
impl<A, F: FnOnce<A> + ?Sized> FnOnce for Box<F> {
    // ...
}
impl<A, F: FnMut<A> + ?Sized> FnMut for Box<F> {
    // ...
}
impl<A, F: Fn<A> + ?Sized> Fn for Box<F> {
    // ...
}
```

## Usage

`Box` can be used almost transparently. You can even use `Box<dyn FnOnce>` now.

```rust
#![feature(boxed_closure_impls)]

fn main() {
    let resource = "hello".to_owned();
    // Create a boxed once-callable closure
    let f: Box<dyn FnOnce(&i32)> = Box::new(|x| {
        let s = resource;
        println!("{}", x);
        println!("{}", s);
    });

    // Call it
    f(&42);
}
```

## The reason for instability

This is unstable because of the first impl.

It would have been easy if we're allowed to tighten the bound:

```rust,ignore
impl<A, F: FnMut<A> + ?Sized> FnOnce for Box<F> {
    // ...
}
```

However, `Box<dyn FnOnce()>` drops out of the modified impl.
To rescue this, we had had a temporary solution called [`fnbox`][fnbox].

[fnbox]: library-features/fnbox.html

Unfortunately, due to minor coherence reasons, `fnbox` and
`FnOnce for Box<impl FnMut>` had not been able to coexist.
We had preferred `fnbox` for the time being.

Now, as [`unsized_locals`][unsized_locals] is implemented, we can just write the
original impl:

[unsized_locals]: language-features/unsized-locals.html

```rust,ignore
impl<A, F: FnOnce<A> + ?Sized> FnOnce for Box<F> {
    type Output = <F as FnOnce<A>>::Output;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        // *self is an unsized rvalue
        <F as FnOnce<A>>::call_once(*self, args)
    }
}
```

However, since `unsized_locals` is a very young feature, we're careful about
this `FnOnce` impl now.

There's another reason for instability: for compatibility with `fnbox`,
we currently allow specialization of the `Box<impl FnOnce>` impl:

```rust,ignore
impl<A, F: FnOnce<A> + ?Sized> FnOnce for Box<F> {
    type Output = <F as FnOnce<A>>::Output;

    // we have "default" here
    default extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        <F as FnOnce<A>>::call_once(*self, args)
    }
}
```

This isn't what we desire in the long term.
