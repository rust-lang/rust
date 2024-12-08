Attempt was made to import an unimportable value. This can happen when trying
to import a method from a trait.

Erroneous code example:

```compile_fail,E0253
mod foo {
    pub trait MyTrait {
        fn do_something();
    }
}

use foo::MyTrait::do_something;
// error: `do_something` is not directly importable

fn main() {}
```

It's invalid to directly import methods belonging to a trait or concrete type.
