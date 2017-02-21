# Visibility and Privacy

These two terms are often used interchangeably, and what they are attempting to
convey is the answer to the question "Can this item be used at this location?"

Rust's name resolution operates on a global hierarchy of namespaces. Each level
in the hierarchy can be thought of as some item. The items are one of those
mentioned above, but also include external crates. Declaring or defining a new
module can be thought of as inserting a new tree into the hierarchy at the
location of the definition.

To control whether interfaces can be used across modules, Rust checks each use
of an item to see whether it should be allowed or not. This is where privacy
warnings are generated, or otherwise "you used a private item of another module
and weren't allowed to."

By default, everything in Rust is *private*, with two exceptions: Associated
items in a `pub` Trait are public by default; Enum variants
in a `pub` enum are also public by default. When an item is declared as `pub`,
it can be thought of as being accessible to the outside world. For example:

```
# fn main() {}
// Declare a private struct
struct Foo;

// Declare a public struct with a private field
pub struct Bar {
    field: i32,
}

// Declare a public enum with two public variants
pub enum State {
    PubliclyAccessibleState,
    PubliclyAccessibleState2,
}
```

With the notion of an item being either public or private, Rust allows item
accesses in two cases:

1. If an item is public, then it can be used externally through any of its
   public ancestors.
2. If an item is private, it may be accessed by the current module and its
   descendants.

These two cases are surprisingly powerful for creating module hierarchies
exposing public APIs while hiding internal implementation details. To help
explain, here's a few use cases and what they would entail:

* A library developer needs to expose functionality to crates which link
  against their library. As a consequence of the first case, this means that
  anything which is usable externally must be `pub` from the root down to the
  destination item. Any private item in the chain will disallow external
  accesses.

* A crate needs a global available "helper module" to itself, but it doesn't
  want to expose the helper module as a public API. To accomplish this, the
  root of the crate's hierarchy would have a private module which then
  internally has a "public API". Because the entire crate is a descendant of
  the root, then the entire local crate can access this private module through
  the second case.

* When writing unit tests for a module, it's often a common idiom to have an
  immediate child of the module to-be-tested named `mod test`. This module
  could access any items of the parent module through the second case, meaning
  that internal implementation details could also be seamlessly tested from the
  child module.

In the second case, it mentions that a private item "can be accessed" by the
current module and its descendants, but the exact meaning of accessing an item
depends on what the item is. Accessing a module, for example, would mean
looking inside of it (to import more items). On the other hand, accessing a
function would mean that it is invoked. Additionally, path expressions and
import statements are considered to access an item in the sense that the
import/expression is only valid if the destination is in the current visibility
scope.

Here's an example of a program which exemplifies the three cases outlined
above:

```
// This module is private, meaning that no external crate can access this
// module. Because it is private at the root of this current crate, however, any
// module in the crate may access any publicly visible item in this module.
mod crate_helper_module {

    // This function can be used by anything in the current crate
    pub fn crate_helper() {}

    // This function *cannot* be used by anything else in the crate. It is not
    // publicly visible outside of the `crate_helper_module`, so only this
    // current module and its descendants may access it.
    fn implementation_detail() {}
}

// This function is "public to the root" meaning that it's available to external
// crates linking against this one.
pub fn public_api() {}

// Similarly to 'public_api', this module is public so external crates may look
// inside of it.
pub mod submodule {
    use crate_helper_module;

    pub fn my_method() {
        // Any item in the local crate may invoke the helper module's public
        // interface through a combination of the two rules above.
        crate_helper_module::crate_helper();
    }

    // This function is hidden to any module which is not a descendant of
    // `submodule`
    fn my_implementation() {}

    #[cfg(test)]
    mod test {

        #[test]
        fn test_my_implementation() {
            // Because this module is a descendant of `submodule`, it's allowed
            // to access private items inside of `submodule` without a privacy
            // violation.
            super::my_implementation();
        }
    }
}

# fn main() {}
```

For a Rust program to pass the privacy checking pass, all paths must be valid
accesses given the two rules above. This includes all use statements,
expressions, types, etc.

## Re-exporting and Visibility

Rust allows publicly re-exporting items through a `pub use` directive. Because
this is a public directive, this allows the item to be used in the current
module through the rules above. It essentially allows public access into the
re-exported item. For example, this program is valid:

```
pub use self::implementation::api;

mod implementation {
    pub mod api {
        pub fn f() {}
    }
}

# fn main() {}
```

This means that any external crate referencing `implementation::api::f` would
receive a privacy violation, while the path `api::f` would be allowed.

When re-exporting a private item, it can be thought of as allowing the "privacy
chain" being short-circuited through the reexport instead of passing through
the namespace hierarchy as it normally would.
