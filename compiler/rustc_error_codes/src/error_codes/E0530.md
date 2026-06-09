A binding shadowed something it shouldn't.

A match arm or a variable has a name that is already used by
something else, e.g.

* struct name
* enum variant
* static
* associated constant

This error may also happen when an enum variant *with fields* is used
in a pattern, but without its fields.

```compile_fail
enum Enum {
    WithField(i32)
}

use Enum::*;
match WithField(1) {
    WithField => {} // error: missing (_)
}
```

Match bindings cannot shadow statics:

```compile_fail,E0530
static TEST: i32 = 0;

let r = 123;
match r {
    TEST => {} // error: name of a static
}
```

Fixed examples:

```
static TEST: i32 = 0;

let r = 123;
match r {
    some_value => {} // ok!
}
```

or

```
const TEST: i32 = 0; // const, not static

let r = 123;
match r {
    TEST => {} // const is ok!
    other_values => {}
}
```
