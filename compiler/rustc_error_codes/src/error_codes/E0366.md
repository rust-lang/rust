An attempt was made to implement `Drop` on a concrete specialization of a
generic type. An example is shown below:

```compile_fail,E0366
struct Foo<T> {
    t: T
}

impl Drop for Foo<u32> {
    fn drop(&mut self) {}
}
```

This code is not legal: it is not possible to specialize `Drop` to a subset of
implementations of a generic type. One workaround for this is to wrap the
generic type, as shown below:

```
struct Foo<T> {
    t: T
}

struct Bar {
    t: Foo<u32>
}

impl Drop for Bar {
    fn drop(&mut self) {}
}
```
