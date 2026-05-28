Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

Erroneous code example:

```compile_fail,E0252
use foo::baz;
use bar::baz; // error, do `use bar::baz as quux` instead

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```

You can use aliases in order to fix this error. Example:

```
use foo::baz as foo_baz;
use bar::baz; // ok!

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```

Or you can reference the item with its parent:

```
use bar::baz;

fn main() {
    let x = foo::baz; // ok!
}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```
