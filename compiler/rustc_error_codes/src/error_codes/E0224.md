A trait object was declared with no traits.

Erroneous code example:

```compile_fail,E0224
type Foo = dyn 'static +;
```

Rust does not currently support this.

To solve, ensure that the trait object has at least one trait:

```
type Foo = dyn 'static + Copy;
```
