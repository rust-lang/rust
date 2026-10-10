#### Note: this error code is no longer emitted by the compiler.

In-band lifetimes cannot be used in `fn`/`Fn` syntax.

Erroneous code examples:

```ignore (feature got removed)
#![feature(in_band_lifetimes)]

fn foo(x: fn(&'a u32)) {} // error!

fn bar(x: &Fn(&'a u32)) {} // error!

fn baz(x: fn(&'a u32), y: &'a u32) {} // error!

struct Foo<'a> { x: &'a u32 }

impl Foo<'a> {
    fn bar(&self, x: fn(&'a u32)) {} // error!
}
```

Lifetimes used in `fn` or `Fn` syntax must be explicitly
declared using `<...>` binders. For example:

```
fn foo<'a>(x: fn(&'a u32)) {} // ok!

fn bar<'a>(x: &Fn(&'a u32)) {} // ok!

fn baz<'a>(x: fn(&'a u32), y: &'a u32) {} // ok!

struct Foo<'a> { x: &'a u32 }

impl<'a> Foo<'a> {
    fn bar(&self, x: fn(&'a u32)) {} // ok!
}
```
