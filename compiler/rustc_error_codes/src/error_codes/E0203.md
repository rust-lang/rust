Having multiple relaxed default bounds is unsupported.

Erroneous code example:

```compile_fail,E0203
struct Bad<T: ?Sized + ?Send>{
    inner: T
}
```

Here the type `T` cannot have a relaxed bound for multiple default traits
(`Sized` and `Send`). This can be fixed by only using one relaxed bound.

```
struct Good<T: ?Sized>{
    inner: T
}
```
