#### Note: this error code is no longer emitted by the compiler.

An attempt was made to access an associated constant through either a generic
type parameter or `Self`. This is not supported yet. An example causing this
error is shown below:

```
trait Foo {
    const BAR: f64;
}

struct MyStruct;

impl Foo for MyStruct {
    const BAR: f64 = 0f64;
}

fn get_bar_bad<F: Foo>(t: F) -> f64 {
    F::BAR
}
```

Currently, the value of `BAR` for a particular type can only be accessed
through a concrete type, as shown below:

```
trait Foo {
    const BAR: f64;
}

struct MyStruct;

impl Foo for MyStruct {
    const BAR: f64 = 0f64;
}

fn get_bar_good() -> f64 {
    <MyStruct as Foo>::BAR
}
```
