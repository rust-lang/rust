A private type was used in a public type signature.

Erroneous code example:

```compile_fail,E0446
#![deny(private_in_public)]
struct Bar(u32);

mod foo {
    use crate::Bar;
    pub fn bar() -> Bar { // error: private type in public interface
        Bar(0)
    }
}

fn main() {}
```

There are two ways to solve this error. The first is to make the public type
signature only public to a module that also has access to the private type.
This is done by using pub(crate) or pub(in crate::my_mod::etc)
Example:

```
struct Bar(u32);

mod foo {
    use crate::Bar;
    pub(crate) fn bar() -> Bar { // only public to crate root
        Bar(0)
    }
}

fn main() {}
```

The other way to solve this error is to make the private type public.
Example:

```
pub struct Bar(u32); // we set the Bar type public
mod foo {
    use crate::Bar;
    pub fn bar() -> Bar { // ok!
        Bar(0)
    }
}

fn main() {}
```
