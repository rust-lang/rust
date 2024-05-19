A super trait has been added to an auto trait.

Erroneous code example:

```compile_fail,E0568
#![feature(auto_traits)]

auto trait Bound : Copy {} // error!

fn main() {}
```

Since an auto trait is implemented on all existing types, adding a super trait
would filter out a lot of those types. In the current example, almost none of
all the existing types could implement `Bound` because very few of them have the
`Copy` trait.

To fix this issue, just remove the super trait:

```
#![feature(auto_traits)]

auto trait Bound {} // ok!

fn main() {}
```
