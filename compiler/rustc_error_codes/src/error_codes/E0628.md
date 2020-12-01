More than one parameter was used for a generator.

Erroneous code example:

```compile_fail,E0628
#![feature(generators, generator_trait)]

fn main() {
    let generator = |a: i32, b: i32| {
        // error: too many parameters for a generator
        // Allowed only 0 or 1 parameter
        yield a;
    };
}
```

At present, it is not permitted to pass more than one explicit
parameter for a generator.This can be fixed by using
at most 1 parameter for the generator. For example, we might resolve
the previous example by passing only one parameter.

```
#![feature(generators, generator_trait)]

fn main() {
    let generator = |a: i32| {
        yield a;
    };
}
```
