Borrowed data escapes outside of closure.

Erroneous code example:

```compile_fail,E0521
let mut list: Vec<&str> = Vec::new();

let _add = |el: &str| {
    list.push(el); // error: `el` escapes the closure body here
};
```

A type annotation of a closure parameter implies a new lifetime declaration.
Consider to drop it, the compiler is reliably able to infer them.

```
let mut list: Vec<&str> = Vec::new();

let _add = |el| {
    list.push(el);
};
```

See the [Closure type inference and annotation][closure-infere-annotation] and
[Lifetime elision][lifetime-elision] sections of the Book for more details.

[closure-infere-annotation]: https://doc.rust-lang.org/book/ch13-01-closures.html#closure-type-inference-and-annotation
[lifetime-elision]: https://doc.rust-lang.org/reference/lifetime-elision.html
