A used type name is not in scope.

Erroneous code examples:

```compile_fail,E0412
impl Something {} // error: type name `Something` is not in scope

// or:

trait Foo {
    fn bar(N); // error: type name `N` is not in scope
}

// or:

fn foo(x: T) {} // type name `T` is not in scope
```

To fix this error, please verify you didn't misspell the type name, you did
declare it or imported it into the scope. Examples:

```
struct Something;

impl Something {} // ok!

// or:

trait Foo {
    type N;

    fn bar(_: Self::N); // ok!
}

// or:

fn foo<T>(x: T) {} // ok!
```

Another case that causes this error is when a type is imported into a parent
module. To fix this, you can follow the suggestion and use File directly or
`use super::File;` which will import the types from the parent namespace. An
example that causes this error is below:

```compile_fail,E0412
use std::fs::File;

mod foo {
    fn some_function(f: File) {}
}
```

```
use std::fs::File;

mod foo {
    // either
    use super::File;
    // or
    // use std::fs::File;
    fn foo(f: File) {}
}
# fn main() {} // don't insert it for us; that'll break imports
```
