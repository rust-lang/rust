Something other than a module was found in visibility scope.

Erroneous code example:

```compile_fail,E0577,edition2018
pub struct Sea;

pub (in crate::Sea) struct Shark; // error!

fn main() {}
```

`Sea` is not a module, therefore it is invalid to use it in a visibility path.
To fix this error we need to ensure `Sea` is a module.

Please note that the visibility scope can only be applied on ancestors!

```edition2018
pub mod Sea {
    pub (in crate::Sea) struct Shark; // ok!
}

fn main() {}
```
