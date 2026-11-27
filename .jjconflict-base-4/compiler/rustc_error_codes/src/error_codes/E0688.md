#### Note: this error code is no longer emitted by the compiler.

In-band lifetimes were mixed with explicit lifetime binders.

Erroneous code example:

```ignore (feature got removed)
#![feature(in_band_lifetimes)]

fn foo<'a>(x: &'a u32, y: &'b u32) {}   // error!

struct Foo<'a> { x: &'a u32 }

impl Foo<'a> {
    fn bar<'b>(x: &'a u32, y: &'b u32, z: &'c u32) {}   // error!
}

impl<'b> Foo<'a> {  // error!
    fn baz() {}
}
```

In-band lifetimes cannot be mixed with explicit lifetime binders.
For example:

```
fn foo<'a, 'b>(x: &'a u32, y: &'b u32) {}   // ok!

struct Foo<'a> { x: &'a u32 }

impl<'a> Foo<'a> {
    fn bar<'b,'c>(x: &'a u32, y: &'b u32, z: &'c u32) {}    // ok!
}

impl<'a> Foo<'a> {  // ok!
    fn baz() {}
}
```
