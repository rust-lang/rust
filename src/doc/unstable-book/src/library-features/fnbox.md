# `fnbox`

The tracking issue for this feature is [#28796]

[#28796]: https://github.com/rust-lang/rust/issues/28796

------------------------

This had been a temporary alternative to the following impls:

```rust,ignore
impl<A, F> FnOnce for Box<F> where F: FnOnce<A> + ?Sized {}
impl<A, F> FnMut for Box<F> where F: FnMut<A> + ?Sized {}
impl<A, F> Fn for Box<F> where F: Fn<A> + ?Sized {}
```

The impls are parallel to these (relatively old) impls:

```rust,ignore
impl<A, F> FnOnce for &mut F where F: FnMut<A> + ?Sized {}
impl<A, F> FnMut for &mut F where F: FnMut<A> + ?Sized {}
impl<A, F> Fn for &mut F where F: Fn<A> + ?Sized {}
impl<A, F> FnOnce for &F where F: Fn<A> + ?Sized {}
impl<A, F> FnMut for &F where F: Fn<A> + ?Sized {}
impl<A, F> Fn for &F where F: Fn<A> + ?Sized {}
```

Before the introduction of [`unsized_locals`][unsized_locals], we had been unable to provide the former impls. That means, unlike `&dyn Fn()` or `&mut dyn FnMut()` we could not use `Box<dyn FnOnce()>` at that time.

[unsized_locals]: language-features/unsized-locals.html

`FnBox()` is an alternative approach to `Box<dyn FnBox()>` is delegated to `FnBox::call_box` which doesn't need unsized locals. As we now have `Box<dyn FnOnce()>` working, the `fnbox` feature is going to be removed.
