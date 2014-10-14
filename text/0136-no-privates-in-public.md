- Start Date: 2014-06-24
- RFC PR #: [#136](https://github.com/rust-lang/rfcs/pull/136)
- Rust Issue #: [#16463](https://github.com/rust-lang/rust/issues/16463)

# Summary

Require a feature gate to expose private items in public APIs, until we grow the
appropriate language features to be able to remove the feature gate and forbid
it entirely.

# Motivation

Privacy is central to guaranteeing the invariants necessary to write
correct code that employs unsafe blocks. Although the current language
rules prevent a private item from being directly named from outside
the current module, they still permit direct access to private items
in some cases. For example, a public function might return a value of
private type. A caller from outside the module could then invoke this
function and, thanks to type inference, gain access to the private
type (though they still could not invoke public methods or access
public fields). This access could undermine the reasoning of the
author of the module. Fortunately, it is not hard to prevent.

# Detailed design

## Overview

The general idea is that:

 * If an item is declared as public, items referred to in the
   public-facing parts of that item (e.g. its type) must themselves be
   declared as public.

Details follow.

## The rules

These rules apply as long as the feature gate is not enabled. After the feature
gate has been removed, they will apply always.

### When is an item "public"?

Items that are explicitly declared as `pub` are always public. In
addition, items in the `impl` of a trait (not an inherent impl) are
considered public if all of the following conditions are met:

 * The trait being implemented is public.
 * All input types (currently, the self type) of the impl are public.
 * *Motivation:* If any of the input types or the trait is public, it
   should be impossible for an outside to access the items defined in
   the impl. They cannot name the types nor they can get direct access
   to a value of those types.
   
### What restrictions apply to public items?

The rules for various kinds of public items are as follows:

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
   
### Examples

Here are some examples to demonstrate the rules.

#### Struct fields

````
// A private struct may refer to any type in any field.
struct Priv {
    a: Priv,
    b: Pub,
    pub c: Priv
}

enum Vapor<A> { X, Y, Z } // Note that A is not used

// Public fields of a public struct may only refer to public types.
pub struct Item {
    // Private field may reference a private type.
    a: Priv,
    
    // Public field must refer to a public type.
    pub b: Pub,

    // ERROR: Public field refers to a private type.
    pub c: Priv,
    
    // ERROR: Public field refers to a private type.
    // For the purposes of this test, we do not descend into the type,
    // but merely consider the names that appear in type parameters
    // on the type, regardless of usage (or lack thereof) within the type
    // definition itself.
    pub d: Vapor<Priv>,
}

pub struct Pub { ... }
````

#### Methods

```
struct Priv { .. }
pub struct Pub { .. }
pub struct Foo { .. }

impl Foo {
    // Illegal: public method with argument of private type.
    pub fn foo(&self, p: Priv) { .. }
}
```

#### Trait bounds

```
trait PrivTrait { ... }

// Error: type parameter on public item bounded by a private trait.
pub struct Foo<X: PrivTrait> { ... }

// OK: type parameter on private item.
struct Foo<X: PrivTrait> { ... }
```

#### Trait definitions

```
struct PrivStruct { ... }

pub trait PubTrait {
    // Error: private struct referenced from method in public trait
    fn method(x: PrivStruct) { ... }
}

trait PrivTrait {
    // OK: private struct referenced from method in private trait 
    fn method(x: PrivStruct) { ... }
}
```

#### Implementations

To some extent, implementations are prevented from exposing private
types because their types must match the trait. However, that is not
true with generics.

```
pub trait PubTrait<T> {
    fn method(t: T);
}

struct PubStruct { ... }

struct PrivStruct { ... }

impl PubTrait<PrivStruct> for PubStruct {
           // ^~~~~~~~~~ Error: Private type referenced from impl of
           //            public trait on a public type. [Note: this is
           //            an "associated type" here, not an input.]

    fn method(t: PrivStruct) {
              // ^~~~~~~~~~ Error: Private type in method signature.
              //
              // Implementation note. It may not be a good idea to report
              // an error here; I think private types can only appear in
              // an impl by having an associated type bound to a private
              // type.
    }
}
```

#### Type aliases

Note that the path to the public item does not have to be private.

```
mod impl {
    pub struct Foo { ... }
}
pub type Bar = self::impl::Foo;
```

### Negative examples

The following examples should fail to compile under these rules.

#### Non-public items referenced by a pub use

These examples are illegal because they use a `pub use` to re-export
a private item:

````
struct Item { ... }
pub mod module {
    // Error: Item is not declared as public, but is referenced from
    // a `pub use`.
    pub use Item;
}
````

````
struct Foo { ... }
// Error: Non-public item referenced by `pub use`.
pub use Item = Foo;
````

If it was desired to have a private name that is publicly "renamed" using a pub
use, that can be achieved using a module:

```
mod impl {
    pub struct ItemPriv;
}
pub use Item = self::impl::ItemPriv;
```

# Drawbacks

Adds a (temporary) feature gate.

Requires some existing code to opt-in to the feature gate before
transitioning to a more explicit alternative.

Requires effort to implement.

# Alternatives

If we stick with the status quo, we'll have to resolve several bizarre questions
and keep supporting its behavior indefinitely after 1.0.

Instead of a feature gate, we could just ban these things outright right away,
at the cost of temporarily losing some convenience and a small amount of
expressiveness before the more principled replacement features are implemented.

We could make an exception for private supertraits, as these are not quite as
problematic as the other cases. However, especially given that a more principled
alternative is known (private methods), I would rather not make any exceptions.

The original design of this RFC had a stronger notion of "public"
which also considered whether a public path existed to the item. In
other words, a module `X` could not refer to a public item `Y` from a
submodule `Z`, unless `X` also exposed a public path to `Y` (whether
that be because `Z` was public, or via a `pub use`).  This definition
strengthened the basic guarantee of "private things are only directly
accessible from within the current module" to include the idea that
public functions in outer modules cannot accidentally refer to public
items from inner modules unless there is a public path from the outer
to the inner module.  Unfortunately, these rules were complex to state
concisely and also hard to understand in practice; when an error
occurred under these rules, it was very hard to evaluate whether the
error was legitimate. The newer rules are simpler while still
retaining the basic privacy guarantee.

One important advantage of the earlier approach, and a scenario not
directly addressed in this RFC, is that there may be items which are
declared as public by an inner module but *still* not intended to be
exposed to the world at large (in other words, the items are only
expected to be used within some subtree). A special case of this is
crate-local data. In the older rules, the "intended scope" of privacy
could be somewhat inferred from the existence (or non-existence) of
`pub use` declarations. However, in the author's opinion, this
scenario would be best addressed by making `pub` declarations more
expressive so that the intended scope can be stated directly.

