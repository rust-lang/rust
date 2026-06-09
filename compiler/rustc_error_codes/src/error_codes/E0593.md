You tried to supply an `Fn`-based type with an incorrect number of arguments
than what was expected.

Erroneous code example:

```compile_fail,E0593
fn foo<F: Fn()>(x: F) { }

fn main() {
    // [E0593] closure takes 1 argument but 0 arguments are required
    foo(|y| { });
}
```

You have to provide the same number of arguments as expected by the `Fn`-based
type. So to fix the previous example, we need to remove the `y` argument:

```
fn foo<F: Fn()>(x: F) { }

fn main() {
    foo(|| { }); // ok!
}
```
