Attempted to access a field on a primitive type.

Erroneous code example:

```compile_fail,E0610
let x: u32 = 0;
println!("{}", x.foo); // error: `{integer}` is a primitive type, therefore
                       //        doesn't have fields
```

Primitive types are the most basic types available in Rust and don't have
fields. To access data via named fields, struct types are used. Example:

```
// We declare struct called `Foo` containing two fields:
struct Foo {
    x: u32,
    y: i64,
}

// We create an instance of this struct:
let variable = Foo { x: 0, y: -12 };
// And we can now access its fields:
println!("x: {}, y: {}", variable.x, variable.y);
```

For more information about [primitives] and [structs], take a look at the Book.

[primitives]: https://doc.rust-lang.org/book/ch03-02-data-types.html
[structs]: https://doc.rust-lang.org/book/ch05-00-structs.html
