- Start Date: 2014-06-24
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Make it illegal to expose private items in public APIs.


# Motivation

Seeing private items in the types of public items is weird.

It leads to various "what if" scenarios we need to think about and deal with, 
and it's easier to avoid them altogether.

It's the safe choice for 1.0, because we can liberalize things later if we 
choose to, but not make them more restrictive.

If I see an item mentioned in rust-doc generated documentation, I should be 
able to click it to see its documentation in turn.


# Detailed design

## Overview

The general idea is that:

 * If an item is publicly exposed by a module `module`, items referred to in
   the public-facing parts of that item (e.g. its type) must themselves be 
   public.

 * An item referred to in `module` is considered to be public if it is visible 
   to clients of `module`.

Details follow.


## The rules

An item is considered to be publicly exposed by a module if it is declared `pub`
by that module, or if it is re-exported using `pub use` by that module.

For items which are publicly exposed by a module, the rules are that:

 * If it is a `static` declaration, items referred to in its type must be public.

 * If it is an `fn` declaration, items referred to in its trait bounds, argument
   types, and return type must be public.

 * If it is a `struct` or `enum` declaration, items referred to in its trait 
   bounds and in the types of its `pub` fields must be public.

 * If it is a `type` declaration, items referred to in its definition must be 
   public.

 * If it is a `trait` declaration, items referred to in its super-traits, in the
   trait bounds of its type parameters, and in the signatures of its methods 
   (see `fn` case above) must be public.


## What does "public" mean?

An item `Item` referred to in the module `module` is considered to be public if:

 * The qualified name used by `module` to refer to `Item`, when recursively
   resolved through `use` declarations back to the original declaration of 
   `Item`, resolves along the way to at least one `pub` declaration, whether a
   `pub use` declaration or a `pub` original declaration; and

 * For at least one of the above resolved-to `pub` declarations, all ancestor 
   modules of the declaration, up to the deepest common ancestor module of the
   declaration with `module`, are `pub`.
 
In all other cases, an `Item` referred to in `module` is not considered to be
public, or `module` itself cannot refer to `Item` and the distinction is 
irrelevant.

### Examples

In the following examples, the item `Item` referred to in the module `module`
is considered to be public:

````
pub mod module {
    pub struct Item { ... }
}
````

````
pub struct Item { ... }
pub mod module {
    use Item;
}
````

````
pub mod x {
    pub struct Item { ... }
}
pub mod module {
    use x::Item;
}
````

````
pub mod module {
    pub mod x {
        pub struct Item { ... }
    }
    use self::x::Item;
}
````

````
struct Item { ... }
pub mod module {
    pub use Item;
}
````

````
struct Foo { ... }
pub use Item = Foo;
pub mod module {
    use Item;
}
````

````
struct Foo { ... }
pub use Bar = Foo;
use Item = Bar;
pub mod module {
    use Item;
}
````

````
struct Item { ... }
pub mod x {
    pub use Item;
    pub mod y {
        use x::Item;
        pub mod module {
            use super::Item;
        }
    }
}
````

In the above examples, it is assumed that `module` will refer to `Item` as 
simply `Item`, but the same thing holds true if `module` refrains from importing
`Item` explicitly with a private `use` declaration, and refers to it directly by
qualifying it with a path instead.


In the below examples, the item `Item` referred to in the module `module` is 
*not* considered to be public:

````
pub mod module {
    struct Item { ... }
}
````

````
struct Item { ... }
pub mod module {
    use Item;
}
````

````
mod x {
    pub struct Item { ... }
}
pub mod module {
    use x::Item;
}
````

````
pub mod module {
    use self::x::Item;
    mod x {
        pub struct Item { ... }
    }
}
````

````
struct Item { ... }
pub use Alias = Item;
pub mod x {
    pub use Item;
    pub mod module {
        use Item; // refers to top-level `Item`
    }
}
````


# Drawbacks

Requires effort to implement.

May break existing code.

It may turn out that there are use cases which become inexpressible. I don't
expect this, but if there are, we should consider solutions to them on a
case-by-case basis.


# Alternatives

The alternative is the status quo, and the impact of not doing this is that 
we'll have to live with it forever. *(dramatic music)*


# Unresolved questions

Is this the right set of rules to apply?

Did I describe them correctly in the "Detailed design"?

Did I miss anything? Are there any holes or contradictions?

Is there a simpler, easier, and/or more logical formulation?
