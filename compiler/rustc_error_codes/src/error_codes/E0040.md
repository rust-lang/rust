It is not allowed to manually call destructors in Rust.

Erroneous code example:

```compile_fail,E0040
struct Foo {
    x: i32,
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}

fn main() {
    let mut x = Foo { x: -7 };
    x.drop(); // error: explicit use of destructor method
}
```

It is unnecessary to do this since `drop` is called automatically whenever a
value goes out of scope. However, if you really need to drop a value by hand,
you can use the `std::mem::drop` function:

```
struct Foo {
    x: i32,
}
impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}
fn main() {
    let mut x = Foo { x: -7 };
    drop(x); // ok!
}
```
