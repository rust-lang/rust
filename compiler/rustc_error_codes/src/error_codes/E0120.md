Drop was implemented on a trait, which is not allowed: only structs and
enums can implement Drop.

Erroneous code example:

```compile_fail,E0120
trait MyTrait {}

impl Drop for MyTrait {
    fn drop(&mut self) {}
}
```

A workaround for this problem is to wrap the trait up in a struct, and implement
Drop on that:

```
trait MyTrait {}
struct MyWrapper<T: MyTrait> { foo: T }

impl <T: MyTrait> Drop for MyWrapper<T> {
    fn drop(&mut self) {}
}

```

Alternatively, wrapping trait objects requires something:

```
trait MyTrait {}

//or Box<MyTrait>, if you wanted an owned trait object
struct MyWrapper<'a> { foo: &'a MyTrait }

impl <'a> Drop for MyWrapper<'a> {
    fn drop(&mut self) {}
}
```
