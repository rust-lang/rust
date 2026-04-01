Generics have been used on an auto trait.

Erroneous code example:

```compile_fail,E0567
#![feature(auto_traits)]

auto trait Generic<T> {} // error!
# fn main() {}
```

Since an auto trait is implemented on all existing types, the
compiler would not be able to infer the types of the trait's generic
parameters.

To fix this issue, just remove the generics:

```
#![feature(auto_traits)]

auto trait Generic {} // ok!
# fn main() {}
```
