# Early/Late bound parameters

This section discusses what it means for generic parameters to be early or late bound.

```rust
fn foo<'a, T>(b: &'a T) -> &'a T { b }
//     ^^  ^early bound
//     ^^
//     ^^late bound
```

Generally when referring to an item with generic parameters you must specify a list of generic arguments corresponding to the item's generic parameters. In
some cases it is permitted to elide these arguments but still, implicitly, a set of arguments are provided (i.e. `Vec::default()` desugars to `Vec::<_>::default()`).

For functions this is not necessarily the case, for example if we take the function `foo` from the example above and write the following code:
```rust
fn main() {
    let f = foo::<_>;

    let b = String::new();
    let c = String::new();
    
    f(&b);
    drop(b);
    f(&c);
}
```

This code compiles perfectly fine even though there is no single lifetime that could possibly be specified in `foo::<_>` that would allow for both
the `&b` and `&c` borrows to be used as arguments (note: the `drop(b)` line forces the `&b` borrow to be shorter than the `&c` borrow). This works because
the `'a` lifetime is _late bound_.

A generic parameter being late bound means that when we write `foo::<_>` we do not actually provide an argument for that parameter, instead we wait until _calling_ the function to provide the generic argument. In the above example this means that we are doing something like `f::<'_>(&b);` and `f::<'_>(&c);` (although in practice we do not actually support turbofishing late bound parameters in this manner)

It may be helpful to think of "early bound parameter" or "late bound parameter" as meaning "early provided parameter" and "late provided parameter", i.e. we provide the argument to the parameter either early (when naming the function) or late (when calling it).