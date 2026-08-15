# `virtual-function-elimination`

This option controls whether LLVM runs the Virtual Function Elimination (VFE)
optimization. This optimization in only available with LTO, so this flag can
only be passed if [`-Clto`][Clto] is also passed.

VFE makes it possible to remove functions from vtables that are never
dynamically called by the rest of the code. Without this flag, LLVM makes the
really conservative assumption, that if any function in a vtable is called, no
function that is referenced by this vtable can be removed. With this flag
additional information are given to LLVM, so that it can determine which
functions are actually called and remove the unused functions.

## Limitations

At the time of writing this flag may remove vtable functions too eagerly. One
such example is in this code:

```rust
trait Foo { fn foo(&self) { println!("foo") } }

impl Foo for usize {}

pub struct FooBox(Box<dyn Foo>);

pub fn make_foo() -> FooBox { FooBox(Box::new(0)) }

#[inline]
pub fn f(a: FooBox) { a.0.foo() }
```

In the above code the `Foo` trait is private, so an assumption is made that its
functions can only be seen/called from the current crate and can therefore get
optimized out, if unused. However, with `make_foo` you can produce a wrapped
`dyn Foo` type outside of the current crate, which can then be used in `f`. Due
to inlining of `f`, `Foo::foo` can then be called from a foreign crate. This can
lead to miscompilations.

[Clto]: ../../rustc/codegen-options/index.html#lto
