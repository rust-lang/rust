A yield expression was used outside of the generator literal.

Erroneous code example:

```compile_fail,E0627
#![feature(generators, generator_trait)]

fn fake_generator() -> &'static str {
    yield 1;
    return "foo"
}

fn main() {
    let mut generator = fake_generator;
}
```

The error occurs because keyword `yield` can only be used inside the generator
literal. This can be fixed by constructing the generator correctly.

```
#![feature(generators, generator_trait)]

fn main() {
    let mut generator = || {
        yield 1;
        return "foo"
    };
}
```
