A module cannot be found and therefore, the visibility cannot be determined.

Erroneous code example:

```compile_fail,E0578,edition2018
foo!();

pub (in ::Sea) struct Shark; // error!

fn main() {}
```

Because of the call to the `foo` macro, the compiler guesses that the missing
module could be inside it and fails because the macro definition cannot be
found.

To fix this error, please be sure that the module is in scope:

```edition2018
pub mod Sea {
    pub (in crate::Sea) struct Shark;
}

fn main() {}
```
