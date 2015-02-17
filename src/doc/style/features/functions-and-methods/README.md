% Functions and methods

### Prefer methods to functions if there is a clear receiver. **[FIXME: needs RFC]**

Prefer

```rust
impl Foo {
    pub fn frob(&self, w: widget) { ... }
}
```

over

```rust
pub fn frob(foo: &Foo, w: widget) { ... }
```

for any operation that is clearly associated with a particular
type.

Methods have numerous advantages over functions:
* They do not need to be imported or qualified to be used: all you
  need is a value of the appropriate type.
* Their invocation performs autoborrowing (including mutable borrows).
* They make it easy to answer the question "what can I do with a value
  of type `T`" (especially when using rustdoc).
* They provide `self` notation, which is more concise and often more
  clearly conveys ownership distinctions.

> **[FIXME]** Revisit these guidelines with
> [UFCS](https://github.com/nick29581/rfcs/blob/ufcs/0000-ufcs.md) and
> conventions developing around it.



### Guidelines for inherent methods. **[FIXME]**

> **[FIXME]** We need guidelines for when to provide inherent methods on a type,
> versus methods through a trait or functions.

> **NOTE**: Rules for method resolution around inherent methods are in flux,
> which may impact the guidelines.
