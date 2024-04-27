#### Note: this error code is no longer emitted by the compiler.

You cannot define a struct (or enum) `Foo` that requires an instance of `Foo`
in order to make a new `Foo` value. This is because there would be no way a
first instance of `Foo` could be made to initialize another instance!

Here's an example of a struct that has this problem:

```
struct Foo { x: Box<Foo> } // error
```

One fix is to use `Option`, like so:

```
struct Foo { x: Option<Box<Foo>> }
```

Now it's possible to create at least one instance of `Foo`: `Foo { x: None }`.
