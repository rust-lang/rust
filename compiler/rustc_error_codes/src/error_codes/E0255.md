You can't import a value whose name is the same as another value defined in the
module.

Erroneous code example:

```compile_fail,E0255
use bar::foo; // error: an item named `foo` is already in scope

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```

You can use aliases in order to fix this error. Example:

```
use bar::foo as bar_foo; // ok!

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```

Or you can reference the item with its parent:

```
fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {
    bar::foo(); // we get the item by referring to its parent
}
```
