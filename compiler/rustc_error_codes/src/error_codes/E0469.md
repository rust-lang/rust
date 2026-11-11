A macro listed for import was not found.

Erroneous code example:

```compile_fail,E0469
#[macro_use(drink, be_merry)] // error: imported macro not found
extern crate alloc;

fn main() {
    // ...
}
```

Either the listed macro is not contained in the imported crate, or it is not
exported from the given crate.

This could be caused by a typo. Did you misspell the macro's name?

Double-check the names of the macros listed for import, and that the crate
in question exports them.

A working version would be:

```ignore (cannot-doctest-multicrate-project)
// In some_crate crate:
#[macro_export]
macro_rules! eat {
    ...
}

#[macro_export]
macro_rules! drink {
    ...
}

// In your crate:
#[macro_use(eat, drink)]
extern crate some_crate; //ok!
```
