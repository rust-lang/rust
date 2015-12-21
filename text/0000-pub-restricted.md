- Feature Name: pub_restricted
- Start Date: 2015-12-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Expand the current `pub`/non-`pub` categorization of items with the
ability to say "make this item visible *solely* to a (named) module
tree."

The current `crate` is one such tree, and would be expressed via:
`pub(crate) item`. Other trees can be denoted via a path employed in a
`use` statement, e.g. `pub(a::b) item`, or `pub(super) item`.

# Motivation
[motivation]: #motivation

Right now, if you have a definition for an item `X` that you want to
use in many places in a module tree, you can either 
(1.) define `X` at the root of the tree as a non-`pub` item, or
(2.) you can define `X` as a `pub` item in some submodule
(and import into the root of the module tree via `use`).

But: Sometimes neither of these options is really what you want.

There are scenarios where developers would like an item to be visible
to a particular module subtree (or a whole crate in its entirety), but
it is not possible to move the item's (non-pub) definition to the root
of that subtree (which would be the usual way to expose an item to a
subtree without making it pub).

If the definition of `X` itself needs access to other private items
within a submodule of the tree, then `X` *cannot* be put at the root
of the module tree. Illustration:

```rust
// Intent: `a` exports `I` and `foo`, but nothing else.
pub mod a {
    pub const I: i32 = 3;

    // `semisecret` will be used "many" places within `a`, but
    // is not meant to be exposed outside of `a`.
    fn semisecret(x: i32) -> i32  { use self::b::c::J; x + J }

    pub fn foo(y: i32) -> i32 { semisecret(I) + y }
    pub fn bar(z: i32) -> i32 { semisecret(I) * z }

    mod b {
        mod c {
            const J: i32 = 4; // J is meant to be hidden from the outside world.
        }
    }
}
```

(Note: the `pub mod a` is meant to be at the root of some crate.)

The latter code fails to compile, due to the privacy violation where
the body of `fn semisecret` attempts to access `a::b::c::J`, which
is not visible in the context of `a`.

A standard way to deal with this today is to use the second approach
described above (labelled "(2.)"): move `fn semisecret` down into the place where it can
access `J`, marking `fn semisecret` as `pub` so that it can still be
accessed within the items of `a`, and then re-exporting `semisecret`
as necessary up the module tree.

```rust
// Intent: `a` exports `I` and `foo`, but nothing else.
pub mod a {
    pub const I: i32 = 3;

    // `semisecret` will be used "many" places within `a`, but
    // is not meant to be exposed outside of `a`.
    // (If we put `pub use` here, then *anyone* could access it.)
    use self::b::semisecret;

    pub fn foo(y: i32) -> i32 { semisecret(I) + y }
    pub fn bar(z: i32) -> i32 { semisecret(I) * z }

    mod b {
        pub use self::c::semisecret;
        mod c {
            const J: i32 = 4; // J is meant to be hidden from the outside world.
            pub fn semisecret(x: i32) -> i32  { x + J }
        }
    }
}
```

This works, but there is a serious issue with it: One cannot easily
tell exactly how "public" `fn semisecret` is. In particular,
understanding who can access `semisecret` requires reasoning about
(1.) all of the `pub use`'s (aka re-exports) of `semisecret`, and
(2.) the `pub`-ness of every module in a path leading to `fn
semisecret` or one of its re-exports.

This RFC seeks to remedy the above problem via two main changes.

  1. Give the user a way to explicitly restrict the intended scope
     of where a `pub`-licized item can be used.

  2. Modify the privacy rules so that `pub`-restricted items cannot be
     used nor re-exported outside of their respective restricted areas.

## Impact

This difficulty in reasoning about the "publicness" of a name is not
just a problem for users; it also complicates efforts within the
compiler to verify that a surface API for a type does not itself use
or expose any private names.

[There][18241] are [a][28325] number [of][28450] bugs [filed][28514] against
[privacy][29668] checking; some are simply
implementation issues, but the comment threads in the issues make it
clear that in some cases, different people have very different mental
models about how privacy interacts with aliases (e.g. `type`
declarations) and re-exports.

In theory, we can add the changes of this RFC without breaking any old
code. (That is, in principle the only affected code is that for item
definitions that use `pub(restriction)`. This limited addition would
still provide value to users in their reasoning about the visibility
of such items.)

In practice, I expect that as part of the implementation of this RFC,
we will probably fix pre-existing bugs in the parts of privacy
checking verifying that surface API's do not use or expose private
names.

Important: No such fixes to such pre-existing bugs are being
concretely proposed by this RFC; I am merely musing that by adding a
more expressive privacy system, we will open the door to fix bugs
whose exploits, under the old system, were the only way to express
certain patterns of interest to developers.

<!-- Trait re-exports fail due to privacy of containing module -->
[18241]: https://github.com/rust-lang/rust/issues/18241

<!-- Rules governing references to private types in public APIs not enforced in impls -->
[28325]: https://github.com/rust-lang/rust/issues/28325

<!-- Type alias can be used to bypass privacy check -->
[28450]: https://github.com/rust-lang/rust/issues/28450

<!-- Private trait's methods reachable through a public supertrait -->
[28514]: https://github.com/rust-lang/rust/issues/28514

<!-- Non-exported type in exported type signature does not error -->
[29668]: https://github.com/rust-lang/rust/issues/29668

<!-- Ban private items in public APIs -->
[RFC 136]: https://github.com/rust-lang/rfcs/blob/master/text/0136-no-privates-in-public.md

<!-- The original definition of public items was based on reachability rather than simply checking whether the item is declared pub or not. The RFC also did not describe how privacy and impl definitions are related. -->
[RFC amendment 200]: https://github.com/rust-lang/rfcs/pull/200


# Detailed design
[design]: #detailed-design

The main problem identified in the [motivation][] section is this:

From an module-internal definition like
```rust
pub mod a { [...] mod b { [...] pub fn semisecret(x: i32) -> i32  { x + J } [...] } }
```
one cannot readily tell exactly how "public" the `fn semisecret` is meant to be.

As already stated, this RFC seeks to remedy the above problem via two
main changes.

  1. Give the user a way to explicitly restrict the intended scope
     of where a `pub`-licized item can be used.

  2. Modify the privacy rules so that `pub`-restricted items cannot be
     used nor re-exported outside of their respective restricted areas.

## Syntax

The new feature is to restrict the scope by adding the module subtree
(which acts as the restricted area) in parentheses after the `pub`
keyword, like so:

```rust
pub(a::b::c) item;
```

The path in the restriction is resolved just like a `use` statement: it
is resolved absolutely, from the crate root.

Just like `use` statements, one can also write relative paths, by
starting them with `self` or a sequence of `super`'s.

```rust
pub(super::super) item;
// or
pub(self) item; // (semantically equiv to no `pub`; see below)
```

In addition to the forms analogous to `use`, there is one new form:

```rust
pub(crate) item;
```

In other words, the grammar is changed like so:

old:
```
VISIBILITY ::= <empty> | `pub`
```

new:
```
VISIBILITY ::= <empty> | `pub` | `pub` `(` USE_PATH `)` | `pub` `(` `crate` `)`
```

One can use these `pub(restriction)` forms anywhere that one can
currently use `pub`. In particular, one can use them on item
defintions, methods in an impl, the fields of a struct
definition, and on `pub use` re-exports.

## Semantics

The meaning of `pub(restriction)` is as follows: The definition of
every item, method, field, or name (e.g. a re-export) is associated
with a restriction.

A restriction is either: the universe of all crates (aka
"unrestricted"), the current crate, or an absolute path to a module
sub-hierarchy in the current crate. A restricted thing cannot be
directly "used" in source code outside of its restricted area.  (The
term "used" here is meant to cover both direct reference in the
source, and also implicit reference as the inferred type of an
expression or pattern.)

 * `pub` written with no explicit restriction means that there is no
   restriction, or in other words, the restriction is the universe of
   all crates.

 * `pub(crate)` means that the restriction is the current crate.

 * `pub(<path>)` means that the restriction is the module
   sub-hierarchy denoted by `<path>`, resolved in the context of the
   occurrence of the `pub` modifier. (This is to ensure that `super`
   and `self` make sense in such paths.)

As noted above, the definition means that `pub(self) item` is the same
as if one had written just `item`.

 * The main reason to support this level of generality (which is
   otherwise just "redundant syntax") is macros: one can write a macro
   that expands to `pub($arg) item`, and a macro client can pass in
   `self` as the `$arg` to get the effect of a non-pub definition.

NOTE: even if the restriction of an item or name indicates that it is
accessible in some context, it may still be impossible to reference
it. In particular, we will still keep our existing rules regarding
`pub` items defined in non-`pub` modules; such items would have no
restriction, but still may be inaccessible if they are not re-exported in
some manner.

## Revised Example
[revised]: #revised-example

In the running example, one could instead write:

```rust
// Intent: `a` exports `I` and `foo`, but nothing else.
pub mod a {
    pub const I: i32 = 3;

    // `semisecret` will be used "many" places within `a`, but
    // is not meant to be exposed outside of `a`.
    // (`pub use` would be *rejected*; see Note 1 below)
    use self::b::semisecret;

    pub fn foo(y: i32) -> i32 { semisecret(I) + y }
    pub fn bar(z: i32) -> i32 { semisecret(I) * z }

    mod b {
        pub(a) use self::c::semisecret;
        mod c {
            const J: i32 = 4; // J is meant to be hidden from the outside world.

            // `pub(a)` means "usable within hierarchy of `mod a`, but not
            // elsewhere."
            pub(a) fn semisecret(x: i32) -> i32  { x + J }
        }
    }
}
```

Note 1: The compiler would reject the variation of the above written
as:

```rust
pub mod a { [...] pub use self::b::semisecret; [...] }
```

because `pub(a) fn semisecret` says that it cannot be used outside of
`a`, and therefore it be incorrect (or at least useless) to reexport
`semisecret` outside of `a`.

Note 2: The most direct interpretation of the rules here leads me to
conclude that `b`'s re-export of `semisecret` needs to be restricted
to `a` as well. However, it may be possible to loosen things so that
the re-export could just stay as `pub` with no extra restriction; see
discussion of "IRS:PUNPM" in Unresolved Questions.

This richer notion of privacy does offer us some other ways to
re-write the running example; instead of defining `fn semisecret`
within `c` so that it can access `J`, we might instead expose `J` to
`mod b` and then put `fn semisecret`, like so:

```rust
pub mod a {
    [...]
    mod b {
        use self::c::J;
        pub(a) fn semisecret(x: i32) -> i32  { x + J }
        mod c {
            pub(b) const J: i32 = 4;
        }
    }
}
```

(This RFC takes no position on which of the above two structures is
"better"; a toy example like this does not provide enough context to
judge.)

## Restrictions
[restrictions]: #restrictions

Lets discuss what the restrictions actually mean.

Some basic definitions: An item is just as it is declared in the Rust
reference manual: a component of a crate, located at a fixed path
(potentially at the "outermost" anonymous module) within the module
tree of the crate.

Every item can be thought of as having some hidden implementation
component(s) along with an exposed surface API.

So, for example, in `pub fn foo(x: Input) -> Output { Body }`, the
surface of `foo` includes `Input` and `Output`, while the `Body` is
hidden.

The pre-existing privacy rules (both prior to and after this RFC) try
to enforce two things: (1.) when a item references a path, all of the
names on that path need to be visible (in terms of privacy) in the
referencing context and, (2.) private items should not be exposed in
the surface of public API's.

 * I am using the term "surface" rather than "signature" deliberately,
   since I think the term "signature" is too broad to be used to
   accurately describe the current semantics of rustc. See my recent
   [Surface blog post][] for further discussion.

[Surface blog post]: http://blog.pnkfx.org/blog/2015/12/19/signatures-and-surfaces-thoughts-on-privacy-versus-dependency/

This RFC is expanding the scope of (2.) above, so that the rules are now:

 1. when a item references a path (in its implementation or in its
    signature), all of the names on that path must be visible in the
    referencing context.

 2. items *restricted* to an area R should not be exposed in the
    surface API of names or items that can themselves be exported
    beyond R. (Privacy is now a special case of this more general
    notion.)

    For convenience, it is legal to declare a field (or inherent
    method) with a strictly larger area of restriction than its
    `self`. See discussion in the [examples][parts-more-public-than-whole].

In principle, validating (1.) can be done via the pre-existing privacy
code. (However, it may make sense to do it by mapping each name to its
associated restriction; I don't think that will change the outcome,
but it might make the checking code simpler. But I am not an expert on
the current state of the privacy checking code.)

Validating (2.) requires traversing the surface API for each item and
comparing the restriction for every reference to the restriction of
the item itself.

## Trait methods

Currently, trait associated item syntax carries no `pub` modifier.

A question arises when trying to apply the terminology of this RFC:
are trait associated items implicitly `pub`, in the sense that they
are unrestricted?

The simple answer is: No, associated items are not implicitly `pub`;
at least, not in general. (They are not in general implicitly `pub`
today either, as discussed in [RFC 136][when public (RFC 136)].)
(If they were implictly `pub`, things would be difficult; further
discussion in attached [appendix][associated items digression].)

[when public (RFC 136)]: https://github.com/rust-lang/rfcs/blob/master/text/0136-no-privates-in-public.md#when-is-an-item-public

However, since this RFC is introducing multiple kinds of `pub`, we
should address the topic of what *is* the `pub`-ness of associated
items.

 * When analyzing a trait definition, then associated items should be
   considered to inherit the `pub`-ness, if any, of their defining
   trait.

   We want to make sure that this code continues to work:

   ```rust
   mod a {
       struct S(String);
       trait Trait {
           fn make_s(&self) -> S; // referencing `S` is ok, b/c `Trait` is not `pub`
       }
   }
   ```

   And under this RFC, we now allow this as well:

   ```rust
   mod a {
       struct S(String);
       mod b {
           pub(a) trait Trait {
               fn mk_s(&self) -> ::a::S;
               // referencing `::a::S` is ok, b/c `Trait` is restricted to `::a`
           }
       }
       use self::b::Trait;
   }
   ```

   Note that in stable Rust today, it is an error to declare the latter trait
   within `mod b` as non-`pub` (since the `use self::b::Trait` would be
   referencing a private item),
   *and* in the Rust nightly channel it is a warning to declare it
   as `pub trait Trait { ... }`.

   The point of this RFC is to give users a sensible way to declare
   such traits within `b`, without allowing them to be exposed outside
   of `a`.

 * When analyzing an `impl Trait for Type`, there may be distinct
   restrictions assigned to the `Trait` and the `Type`. However,
   since both the `Trait` and the `Type` must be visible in the
   context of the module where the `impl` occurs, there should
   be a subtree relationship between the two restrictions; in other
   words, one restriction should be less than (or equal to) the other.

   So just use the minimum of the two restrictions when analyzing
   the right-hand sides of the associated items in the impl.

   Note: I am largely adopting this rule in an attempt to be
   consistent with [RFC 136][when public (RFC 136)]. I invite
   discussion of whether this rule actually makes sense as phrased
   here.

## More examples!
[examples]: #more-examples

These examples meant to explore the syntax a bit. They are *not* meant
to provide motivation for the feature (i.e. I am not claiming that the
feature is making this code cleaner or easier to reason about).

### Impl item example
[impl item example]: #impl-item-example

```rust
pub struct S;

mod a {
    pub fn call_foo(s: &S) { s.foo(); }

    impl S {
        pub(a) fn foo(&self) { println!("only callable within `a`"); }
    }
}

fn rejected(s: &S) {
    s.foo(); //~ ERROR: `S::foo` not visible outside of module `a`
}
```

(You may be wondering: "Could we move that `impl S` out to the
top-level, out of `mod a`?" Well ... see discussion in the
[unresolved questions][def-outside-restriction].)

### Restricting fields example
[restricting fields example]: #restricting-fields-example

```rust
mod a {
    #[derive(Default)]
    struct Priv(i32);

    pub mod b {
        use a::Priv as Priv_a;

        #[derive(Default)]
        pub struct F {
            pub    x: i32,
                   y: Priv_a,
            pub(a) z: Priv_a,
        }

        #[derive(Default)]
        pub struct G(pub i32, Priv_a, pub(a) Priv_a);

        // ... accesses to F.{x,y,z} ...
        // ... accesses to G.{0,1,2} ...
    }
    // ... accesses to F.{x,z} ...
    // ... accesses to G.{0,2} ...
}

mod k {
    use a::b::{F, G};
    // ... accesses to F and F.x ...
    // ... accesses to G and G.0 ...
}
```


### Fields and inherent methods more public than self
[parts-more-public-than-whole]: #fields-and-inherent-methods-more-public-than-self

In Rust today, one can write

```rust
mod a { struct X { pub y: i32, } }
```

This RFC was crafted to say that fields and inherent methods
can have an associated restriction that is larger than the restriction
of its `self`. This was both to keep from breaking the above
code, and also because it would be annoying to be forced to write:

```rust
mod a { struct X { pub(a) y: i32, } }
```

(This RFC is not an attempt to resolve things like
[Rust Issue 30079][30079]; the decision of how to handle that issue
can be dealt with orthogonally, in my opinion.)

[30079]: https://github.com/rust-lang/rust/issues/30079


So, under this RFC, the following is legal:

```rust
mod a {
    pub use self::b::stuff_with_x;
    mod b {
        struct X { pub y: i32, pub(a) z: i32 }
        mod c {
            impl super::X {
                pub(c) fn only_in_c(&mut self) { self.y += 1; }

                pub fn callanywhere(&mut self) {
                    self.only_in_c();
                    println!("X.y is now: {}", self.y);
                }
            }
        }
        pub fn stuff_with_x() {
            let mut x = X { y: 10, z: 20};
            x.callanywhere();
        }
    }
}
```

In particular:

 * It is okay that the fields `y` and `z` and the inherent method
   `fn callanywhere` are more publicly visible than `X`.

   (Just because we declare something `pub` does not mean it will
    actually be *possible* to reach it from arbitrary contexts. Whether
    or not such access is possible will depend on many things, including
    but not limited to the restriction attached and also future decisions
    about issues like [issue 30079][30079].)

 * We are allowed to restrict an inherent method, `fn only_in_c`, to
   a subtree of the module tree where `X` is itself visible.

### Re-exports

Here is an example of a `pub use` re-export using the new
feature, including both correct and invalid uses of the extended form.

```rust
mod a {
    mod b {
        pub(a) struct X { pub y: i32, pub(a) z: i32 } // restricted to `mod a` tree
        mod c {
            pub mod d {
                pub(super) use a::b::X as P; // ok: a::b::c is submodule of `a`
            }

            fn swap_ok(x: d::P) -> d::P { // ok: `P` accessible here
                X { z: x.y, y: x.z }
            }
        }

        fn swap_bad(x: c::d::P) -> c::d::P { //~ ERROR: `c::d::P` not visible outside `a::b::c`
            X { z: x.y, y: x.z }
        }

        mod bad {
            pub use super::X; //~ ERROR: `X` cannot be reexported outside of `a`
        }
    }

    fn swap_ok2(x: X) -> X { // ok: `X` accessible from `mod a`.
        X { z: x.y, y: x.z }
    }
}
```

### Crate restricted visibility

This is a concrete illusration of how one might use the `pub(crate) item` form,
(which is perhaps quite similar to Java's default "package visibility").

Crate `c1`:

```rust
pub mod a {
    struct Priv(i32);

    pub(crate) struct R { pub y: i32, z: Priv } // ok: field allowed to be more public
    pub        struct S { pub y: i32, z: Priv }

    pub fn to_r_bad(s: S) -> R { ... } //~ ERROR: `R` restricted solely to this crate

    pub(crate) fn to_r(s: S) -> R { R { y: s.y, z: s.z } } // ok: restricted to crate
}

use a::{R, S}; // ok: `a::R` and `a::S` are both visible

pub use a::R as ReexportAttempt; //~ ERROR: `a::R` restricted solely to this crate
```

Crate `c2`:

```rust
extern crate c1;

use c1::a::S; // ok: `S` is unrestricted

use c1::a::R; //~ ERROR: `c1::a::R` not visible outside of its crate
```

## Precedent

When I started on this I was not sure if this form of delimited access
to a particular module subtree had a precedent; the closest thing I
could think of was C++ `friend` modifiers (but `friend` is far more
ad-hoc and free-form than what is being proposed here).

### Scala

It has since been pointed out to me that Scala has scoped access
modifiers `protected[Y]` and `private[Y]`, which specify that access
is provided upto `Y` (where `Y` can be a package, class or singleton
object).

The feature proposed by this RFC appears to be similar in intent to
Scala's scoped access modifiers.

Having said that, I will admit that I am not clear on what
distinction, if any, Scala draws between `protected[Y]` and
`private[Y]` when `Y` is a package, which is the main analogy for our
purposes, or if they just allow both forms as synonyms for
convenience.

(I can imagine a hypothetical distinction in Scala when `Y` is a
class, but my skimming online has not provided insight as to what the
actual distinction is.)

Even if there is some distinction drawn between the two forms in
Scala, I suspect Rust does not need an analogous distinction in it's
`pub(restricted)`

# Drawbacks
[drawbacks]: #drawbacks

Obviously,
`pub(restriction) item` complicates the surface syntax of the language.

 * However, my counter-argument to this drawback is that this feature
   in fact *simplifies* the developer's mental model. It is easier to
   directly encode the expected visibility of an item via
   `pub(restriction)` than to figure out the right concoction via a
   mix of nested `mod` and `pub use` statements. And likewise, it is
   easier to read it too.

Developers may misuse this form and make it hard to access the tasty
innards of other modules.

 * This is true, but I claim it is irrelevant.

   The effect of this change is solely on the visibility of items
   *within* a crate. No rules for inter-crate access change.

   From the perspective of cross-crate development, this RFC changes
   nothing, except that it may lead some crate authors to make some
   things no longer universally `pub` that they were forced to make
   visible before due to earlier limitations. I claim that in such
   cases, those crate authors probably always intended for such items
   to be non-`pub`, but language limitations were forcing their hand.

   As for intra-crate access: My expectation is that an individual
   crate will be made by a team of developers who can work out what
   mutual visibility they want and how it should evolve over time.
   This feature may affect their work flow to some degree, but they
   can choose to either use it or not, based on their own internal
   policies.


# Alternatives
[alternatives]: #alternatives

## Do not extend the language!

 * Change privacy rules and make privacy analysis "smarter"
   (e.g. global reachabiliy analysis)

   The main problem with this approach is that we tried it, and it
   did not work well: The implementation was buggy, and the user-visible
   error messages were hard to understand.

   See discussion when the team was discussing the [public items amendment][]

[public items amendment]: https://github.com/rust-lang/meeting-minutes/blob/master/weekly-meetings/2014-09-16.md#rfc-public-items

 * "Fix" the mental model of privacy (if necessary) without extending
   the language.

   The alternative is bascially saying: "Our existing system is fine; all
   of the problems with it are due to bugs in the implementation"

   I am sympathetic to this response. However, I think it doesn't
   quite hold up. Some users want to be able to define items that are
   exposed outside of their module but still restrict the scope of
   where they can be referenced, as discussed in the [motivation][]
   section, and I do not think the current model can be "fixed" to
   support that use case, at least not without adding some sort of
   global reachability analysis as discussed in the previous bullet.

In addition, these two alternatives do not address the main point
being made in the [motivation][] section: one cannot tell exactly how
"public" a `pub` item is, without working backwards through the module
tree for all of its re-exports.

## Curb your ambitions!

 * Instead of adding support for restricting to arbitrary module
   subtrees, narrow the feature to just `pub(crate) item`, so that one
   chooses either "module private" (by adding no modifier), or
   "universally visible" (by adding `pub`), or "visible to just the
   current crate" (by adding `pub(crate)`).

   This would be somewhat analogous to Java's relatively coarse
   grained privacy rules, where one can choose `public`, `private`,
   `protected`, or the unnamed "package" visibility.

   I am all for keeping the implementation simple. However, the reason
   that we should support arbitrary module subtrees is that doing so
   will enable certain refactorings. Namely, if I decide I want to
   inline the definition for one or more crates `A1`, `A2`, ... into
   client crate `C` (i.e. replacing `extern crate A1;` with an
   suitably defined `mod A1 { ... }`, but I do not want to worry about
   whether doing so will risk future changes violating abstraction
   boundaries that were previously being enforced via `pub(crate)`,
   then I believe allowing `pub(path)` will allow a mechanical tool to
   do the inline refactoring, rewriting each `pub(crate)` as `pub(A1)`
   as necessary.

# Unresolved questions
[unresolved]: #unresolved-questions

## Can definition site fall outside restriction?
[def-outside-restriction]: #can-definition-site-fall-outside-restriction

For example, is it illegal to do the following:

```rust
mod a {
  mod child { }
  mod b { pub(super::child) const J: i32 = 3; }
}
```

Or does it just mean that `J`, despite being defined in `mod b`, is
itself not accessible in `mod b`?

pnkfelix is personally inclined to make this sort of thing illegal,
mainly because he finds it totally unintuitive, but is interested in
hearing counter-arguments. Certainly the earlier [impl item example][]
would look prettier as:

```rust
pub struct S;

impl S {
    pub(a) fn foo(&self) { println!("only callable within `a`"); }
}

mod a {
    pub fn call_foo(s: &S) { s.foo(); }

}

fn rejected(s: &S) {
    s.foo(); //~ ERROR: `S::foo` not visible outside of module `a`
}
```

## Implicit Restriction Satisfaction (IRS:PUNPM)

If a re-export occurs within a non-`pub` module, can we treat it as
implicitly satisfying a restriction to `super` imposed by the item it
is re-exporting?

In particular, the [revised example][revised] included:

```rust
// Intent: `a` exports `I` and `foo`, but nothing else.
pub mod a {
    [...]
    mod b {
        pub(a) use self::c::semisecret;
        mod c { pub(a) fn semisecret(x: i32) -> i32  { x + J } }
    }
}
```

However, since `b` is non-`pub`, its `pub` items and re-exports are
solely accessible via the subhierarchy of its module parent (i.e.,
`mod a`, as long as no entity attempts to re-export them to a braoder
scope.

In other words, in some sense `mod b { pub use item; }` *could*
implicitly satisfy a restriction to `super` imposed by `item` (if we
chose to allow it).

Note: If it were `pub mod b` or `pub(restrict) mod b`, then the above
reasoning would not hold.  Therefore, this discussion is limited to
re-exports from non-`pub` modules.

If we do not allow such implicit restriction satisfaction
for `pub use` re-exports from non-`pub` modules (IRS:PUNPM), then:

```rust
pub mod a {
    [...]
    mod b {
        pub use self::c::semisecret;
        mod c { pub(a) fn semisecret(x: i32) -> i32  { x + J } }
    }
}
```

would be rejected, and one would be expected to write either:

```rust
        pub(super) use self::c::semisecret;
```

or

```rust
        pub(a) use self::c::semisecret;
```


(Side note: I am *not* saying that under IRS:PUNPM, the two forms `pub
use item` and `pub(super) use item` would be considered synonymous,
even in the context of a non-pub module like `mod b`. In particular,
`pub(super) use item` may be imposing a new restriction on the
re-exported name that was not part of its original definition.)

# Appendices

## Associated Items Digression
[associated items digression]: #associated-items-digression

If associated items were implicitly `pub`, in the sense that they are
unrestricted, then that would conflict with the rules imposed by this
RFC, in the sense that the surface API of a non-`pub` trait is
composed of its associated items, and so if all associated items were
implicitly `pub` and unrestricted, then this code would be rejected:

```rust
mod a {
    struct S(String);
    trait Trait {
        fn mk_s(&self) -> S; // is this implicitly `pub` and unrestricted?
    }
    impl Trait for () { fn mk_s(&self) -> S { S(format!("():()")) } }
    impl Trait for i32 { fn mk_s(&self) -> S { S(format!("{}:i32", self)) } }
    pub fn foo(x:i32) -> String { format!("silly{}{}", ().mk_s().0, x.mk_s().0) }
}
```

If associated items were implicitly `pub` and unrestricted, then the
above code would be rejected under direct interpretation of the rules
of this RFC (because `fn make_s` is implicitly unrestricted, but the
surface of `fn make_s` references `S`, a non-`pub` item). This would
be backwards-incompatible (and just darn inconvenient too).

So, to be clear, this RFC is *not* suggesting that associated items be
implicitly `pub` and unrestricted.
