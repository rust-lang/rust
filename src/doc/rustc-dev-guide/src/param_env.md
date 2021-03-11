# Parameter Environment

When working with associated and/or generic items (types, constants,
functions/methods) it is often relevant to have more information about the
`Self` or generic parameters. Trait bounds and similar information is encoded in
the [`ParamEnv`][pe]. Often this is not enough information to obtain things like the
type's `Layout`, but you can do all kinds of other checks on it (e.g. whether a
type implements `Copy`) or you can evaluate an associated constant whose value
does not depend on anything from the parameter environment.

[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html

For example if you have a function

```rust
fn foo<T: Copy>(t: T) { ... }
```

the parameter environment for that function is `[T: Copy]`. This means any
evaluation within this function will, when accessing the type `T`, know about
its `Copy` bound via the parameter environment.

You can get the parameter environment for a `def_id` using the
[`param_env`][query] query. However, this `ParamEnv` can be too generic for
your use case. Using the `ParamEnv` from the surrounding context can allow you
to evaluate more things. For example, suppose we had something the following:

[query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ty_utils/ty/fn.param_env.html

```rust
trait Foo {
    type Assoc;
}

trait Bar { }

trait Baz {
    fn stuff() -> bool;
}

fn foo<T>(t: T)
where
    T: Foo,
    <T as Foo>::Assoc: Bar
{
   bar::<T::Assoc>()
}

fn bar<T: Baz>() {
    if T::stuff() { mep() } else { mop() }
}
```

We may know some things inside `bar` that we wouldn't know if we just fetched
`bar`'s param env because of the `<T as Foo>::Assoc: Bar` bound in `foo`. This
is a contrived example that makes no sense in our existing analyses, but we may
run into similar cases when doing analyses with associated constants on generic
traits or traits with assoc types.

## Bundling

Another great thing about `ParamEnv` is that you can use it to bundle the thing
depending on generic parameters (e.g. a `Ty`) by calling the [`and`][and]
method. This will produce a [`ParamEnvAnd<Ty>`][pea], making clear that you
should probably not be using the inner value without taking care to also use
the [`ParamEnv`][pe].

[and]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.and
[pea]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnvAnd.html
