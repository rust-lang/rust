#### Note: this error code is no longer emitted by the compiler.

You can't import a type or module when the name of the item being imported is
the same as another type or submodule defined in the module.

An example of this error:

```compile_fail
use foo::Bar; // error

type Bar = u32;

mod foo {
    pub mod Bar { }
}

fn main() {}
```
