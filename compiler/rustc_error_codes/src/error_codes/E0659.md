An item usage is ambiguous.

Erroneous code example:

```compile_fail,edition2018,E0659
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use crate::moon::*;
    pub use crate::earth::*;
}

fn main() {
    crate::collider::foo(); // ERROR: `foo` is ambiguous
}
```

This error generally appears when two items with the same name are imported into
a module. Here, the `foo` functions are imported and reexported from the
`collider` module and therefore, when we're using `collider::foo()`, both
functions collide.

To solve this error, the best solution is generally to keep the path before the
item when using it. Example:

```edition2018
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use crate::moon;
    pub use crate::earth;
}

fn main() {
    crate::collider::moon::foo(); // ok!
    crate::collider::earth::foo(); // ok!
}
```
