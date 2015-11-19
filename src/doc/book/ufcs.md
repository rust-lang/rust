% Universal Function Call Syntax

Sometimes, functions can have the same names. Consider this code:

```rust
trait Foo {
    fn f(&self);
}

trait Bar {
    fn f(&self);
}

struct Baz;

impl Foo for Baz {
    fn f(&self) { println!("Baz’s impl of Foo"); }
}

impl Bar for Baz {
    fn f(&self) { println!("Baz’s impl of Bar"); }
}

let b = Baz;
```

If we were to try to call `b.f()`, we’d get an error:

```text
error: multiple applicable methods in scope [E0034]
b.f();
  ^~~
note: candidate #1 is defined in an impl of the trait `main::Foo` for the type
`main::Baz`
    fn f(&self) { println!("Baz’s impl of Foo"); }
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
note: candidate #2 is defined in an impl of the trait `main::Bar` for the type
`main::Baz`
    fn f(&self) { println!("Baz’s impl of Bar"); }
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```

We need a way to disambiguate which method we need. This feature is called
‘universal function call syntax’, and it looks like this:

```rust
# trait Foo {
#     fn f(&self);
# }
# trait Bar {
#     fn f(&self);
# }
# struct Baz;
# impl Foo for Baz {
#     fn f(&self) { println!("Baz’s impl of Foo"); }
# }
# impl Bar for Baz {
#     fn f(&self) { println!("Baz’s impl of Bar"); }
# }
# let b = Baz;
Foo::f(&b);
Bar::f(&b);
```

Let’s break it down.

```rust,ignore
Foo::
Bar::
```

These halves of the invocation are the types of the two traits: `Foo` and
`Bar`. This is what ends up actually doing the disambiguation between the two:
Rust calls the one from the trait name you use.

```rust,ignore
f(&b)
```

When we call a method like `b.f()` using [method syntax][methodsyntax], Rust
will automatically borrow `b` if `f()` takes `&self`. In this case, Rust will
not, and so we need to pass an explicit `&b`.

[methodsyntax]: method-syntax.html

# Angle-bracket Form

The form of UFCS we just talked about:

```rust,ignore
Trait::method(args);
```

Is a short-hand. There’s an expanded form of this that’s needed in some
situations:

```rust,ignore
<Type as Trait>::method(args);
```

The `<>::` syntax is a means of providing a type hint. The type goes inside
the `<>`s. In this case, the type is `Type as Trait`, indicating that we want
`Trait`’s version of `method` to be called here. The `as Trait` part is
optional if it’s not ambiguous. Same with the angle brackets, hence the
shorter form.

Here’s an example of using the longer form.

```rust
trait Foo {
    fn foo() -> i32;
}

struct Bar;

impl Bar {
    fn foo() -> i32 {
        20
    }
}

impl Foo for Bar {
    fn foo() -> i32 {
        10
    }
}

fn main() {
    assert_eq!(10, <Bar as Foo>::foo());
    assert_eq!(20, Bar::foo());
}
```

Using the angle bracket syntax lets you call the trait method instead of the
inherent one.
