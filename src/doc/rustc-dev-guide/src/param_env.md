# Parameter Environment

When working with associated and/or or generic items (types, constants,
functions/methods) it is often relevant to have more information about the
`Self` or generic parameters. Trait bounds and similar information is encoded in
the `ParamEnv`. Often this is not enough information to obtain things like the
type's `Layout`, but you can do all kinds of other checks on it (e.g. whether a
type implements `Copy`) or you can evaluate an associated constant whose value
does not depend on anything from the parameter environment.

For example if you have a function

```rust
fn foo<T: Copy>(t: T) {
}
```

the parameter environment for that function is `[T: Copy]`. This means any
evaluation within this function will, when accessing the type `T`, know about
its `Copy` bound via the parameter environment.

Although you can obtain a valid `ParamEnv` for any item via
`tcx.param_env(def_id)`, this `ParamEnv` can be too generic for your use case.
Using the `ParamEnv` from the surrounding context can allow you to evaluate more
things.

Another great thing about `ParamEnv` is that you can use it to bundle the thing
depending on generic parameters (e.g. a `Ty`) by calling `param_env.and(ty)`.
This will produce a `ParamEnvAnd<Ty>`, making clear that you should probably not
be using the inner value without taking care to also use the `ParamEnv`.
