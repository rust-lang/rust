An item was added on a negative impl.

Erroneous code example:

```compile_fail,E0749
# #![feature(negative_impls)]
trait MyTrait {
    type Foo;
}

impl !MyTrait for u32 {
    type Foo = i32; // error!
}
```

Negative impls are not allowed to have any items. Negative impls declare that a
trait is **not** implemented (and never will be) and hence there is no need to
specify the values for trait methods or other items.

One way to fix this is to remove the items in negative impls:

```
# #![feature(negative_impls)]
trait MyTrait {
    type Foo;
}

impl !MyTrait for u32 {}
```
