Attempted to access a method like a field.

Erroneous code example:

```compile_fail,E0615
struct Foo {
    x: u32,
}

impl Foo {
    fn method(&self) {}
}

let f = Foo { x: 0 };
f.method; // error: attempted to take value of method `method` on type `Foo`
```

If you want to use a method, add `()` after it:

```
# struct Foo { x: u32 }
# impl Foo { fn method(&self) {} }
# let f = Foo { x: 0 };
f.method();
```

However, if you wanted to access a field of a struct check that the field name
is spelled correctly. Example:

```
# struct Foo { x: u32 }
# impl Foo { fn method(&self) {} }
# let f = Foo { x: 0 };
println!("{}", f.x);
```
