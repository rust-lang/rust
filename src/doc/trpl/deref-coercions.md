% `Deref` coercions

The standard library provides a special trait, [`Deref`][deref]. It’s normally
used to overload `*`, the dereference operator:

```rust
use std::ops::Deref;

struct DerefExample<T> {
    value: T,
}

impl<T> Deref for DerefExample<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

fn main() {
    let x = DerefExample { value: 'a' };
    assert_eq!('a', *x);
}
```

[deref]: ../std/ops/trait.Deref.html

This is useful for writing custom pointer types. However, there’s a language
feature related to `Deref`: ‘deref coercions’. Here’s the rule: If you have a
type `U`, and it implements `Deref<Target=T>`, values of `&U` will
automatically coerce to a `&T`. Here’s an example:

```rust
fn foo(s: &str) {
    // borrow a string for a second
}

// String implements Deref<Target=str>
let owned = String::from("Hello");

// therefore, this works:
foo(&owned);
```

Using an ampersand in front of a value takes a reference to it. So `owned` is a
`String`, `&owned` is an `&String`, and since `impl Deref<Target=str> for
String`, `&String` will deref to `&str`, which `foo()` takes.

That’s it. This rule is one of the only places in which Rust does an automatic
conversion for you, but it adds a lot of flexibility. For example, the `Rc<T>`
type implements `Deref<Target=T>`, so this works:

```rust
use std::rc::Rc;

fn foo(s: &str) {
    // borrow a string for a second
}

// String implements Deref<Target=str>
let owned = String::from("Hello");
let counted = Rc::new(owned);

// therefore, this works:
foo(&counted);
```

All we’ve done is wrap our `String` in an `Rc<T>`. But we can now pass the
`Rc<String>` around anywhere we’d have a `String`. The signature of `foo`
didn’t change, but works just as well with either type. This example has two
conversions: `Rc<String>` to `String` and then `String` to `&str`. Rust will do
this as many times as possible until the types match.

Another very common implementation provided by the standard library is:

```rust
fn foo(s: &[i32]) {
    // borrow a slice for a second
}

// Vec<T> implements Deref<Target=[T]>
let owned = vec![1, 2, 3];

foo(&owned);
```

Vectors can `Deref` to a slice.

## Deref and method calls

`Deref` will also kick in when calling a method. Consider the following
example.

```rust
struct Foo;

impl Foo {
    fn foo(&self) { println!("Foo"); }
}

let f = &&Foo;

f.foo();
```

Even though `f` is a `&&Foo` and `foo` takes `&self`, this works. That’s
because these things are the same:

```rust,ignore
f.foo();
(&f).foo();
(&&f).foo();
(&&&&&&&&f).foo();
```

A value of type `&&&&&&&&&&&&&&&&Foo` can still have methods defined on `Foo`
called, because the compiler will insert as many * operations as necessary to
get it right. And since it’s inserting `*`s, that uses `Deref`.
