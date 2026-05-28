The `main` function was defined with generic parameters.

Erroneous code example:

```compile_fail,E0131
fn main<T>() { // error: main function is not allowed to have generic parameters
}
```

It is not possible to define the `main` function with generic parameters.
It must not take any arguments.
