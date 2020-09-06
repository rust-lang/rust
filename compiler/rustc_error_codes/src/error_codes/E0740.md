A `union` was declared with fields with destructors.

Erroneous code example:

```compile_fail,E0740
union Test {
    a: A, // error!
}

#[derive(Debug)]
struct A(i32);

impl Drop for A {
    fn drop(&mut self) { println!("A"); }
}
```

A `union` cannot have fields with destructors.
