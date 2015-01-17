- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add a default lifetime bound for object types, so that it is no longer
necessary to write things like `Box<Trait+'static>` or `&'a
(Trait+'a)`. The default will be based on the context in which the
object type appears. Object types that appear underneath a reference
take the lifetime of the innermost reference under which they appear,
and otherwise the default is `'static`.

Examples:

- `&'a &'b SomeTrait` becomes `&'a &'b (SomeTrait+'b)`
- `&'a Box<SomeTrait>` becomes `&'a Box<SomeTrait+'a>`
- `Box<SomeTrait>` becomes `Box<SomeTrait+'static>`
- `Rc<SomeTrait>` becomes `Rc<SomeTrait+'static>`

Cases where the lifetime bound is either given explicitly or can be
inferred from the traits involved are naturally unaffected.

# Motivation

#### Current situation

As described in [RFC 34][34], object types carry a single lifetime
bound. Sometimes, this bound can be inferred based on the traits
involved. Frequently, however, it cannot, and in that case the
lifetime bound must be given explicitly. Some examples of situations
where an error would be reported are as follows:

```rust
struct SomeStruct {
    object: Box<Writer>, // <-- ERROR No lifetime bound can be inferred.
}

struct AnotherStruct<'a> {
    callback: &'a Fn(),  // <-- ERROR No lifetime bound can be inferred.
}
```

Errors of this sort are a [common source of confusion][16948] for new
users (partly due to a poor error message). To avoid errors, those examples
would have to be written as follows:

```rust
struct SomeStruct {
    object: Box<Writer+'static>,
}

struct AnotherStruct<'a> {
    callback: &'a (Fn()+'a),
}
```

Ever since it was introduced, there has been a desire to make this
fully explicit notation more compact for common cases. In practice,
the object bounds are almost always tightly linked to the context in
which the object appears: it is relatively rare, for example, to have
a boxed object type that is not bounded by `'static` or `Send` (e.g.,
`Box<Trait+'a>`). Similarly, it is unusual to have a reference to an
object where the object itself has a distinct bound (e.g., `&'a
(Trait+'b)`). This is not to say these situations *never* arise; as
we'll see below, both of these do arise in practice, but they are
relatively unusual (and in fact there is never a good reason to do
`&'a (Trait+'b)`, though there can be a reason to have `&'a mut
(Trait+'b)`; see ["Detailed Design"](#detailed-design) for full details).

The need for a shorthand is made somewhat more urgent by
[RFC 458][458], which disconnects the `Send` trait from the `'static`
bound. This means that object types now are written `Box<Foo+Send>`
would have to be written `Box<Foo+Send+'static>`.

Therefore, the following examples would require explicit bounds:

```rust
trait Message : Send { }
Box<Message> // ERROR: 'static no longer inferred from `Send` supertrait
Box<Writer+Send> // ERROR: 'static no longer inferred from `Send` bound
```

#### The proposed rule

This RFC proposes a simple rule. If the object type appears within
one or more reference types (`&` or `&mut`), then the default lifetime
bound is taken from the innermost enclosing reference. Otherwise the
default is `'static`.

Here are some statistics showing the frequency of trait references
from three Rust projects. The final column
shows the percentage of uses that would be correctly predicted by this
simple rule.

As these statistics were gathered using `ack` and some simple regular
expressions, they only include cover those cases where an explicit
lifetime bound was required today. In function signatures, lifetime
bounds can always be omitted, and it is impossible to distinguish
`&SomeTrait` from `&SomeStruct` using only a regular
expression. However, we belive that the proposed rule would be
compatible with the existing defaults for function signatures in all
or virtually all cases.

The first table shows the results for objects that appear within a `Box`:

| package | `Box<Trait+Send>` | `Box<Trait+'static>` | `Box<Trait+'other>` |   %  |
|---------|-----------------|--------------------|-------------------|------|
| iron    | 6               | 0                  | 0                 | 100% |
| cargo   | 7               | 0                  | 7                 | 50%  |
| rust    | 53              | 28                 | 20                | 80%  |

Here `rust` refers to both the standard library and rustc. As you can
see, cargo (and rust, specifically libsyntax) both have objects that
encapsulate borrowed references, leading to types
`Box<Trait+'src>`. This pattern is not aided by the current defaults
(though it is also not made any *more* explicit than it already
is). However, this is the minority.

The next table shows the results for references to objects.

| package | `&(Trait+Send)` | `&'a [mut] (Trait+'a)` | `&'a mut (Trait+'b)` |   %  |
|---------|-----------------|----------------------|--------------------|------|
| iron    | 0               | 0                    | 0                  | 100% |
| cargo   | 0               | 0                    | 5                  | 0%   |
| rust    | 1               | 9                    | 0                  | 100% |

As before, the defaults would not help cargo remove its existing
annotations (though they do not get any worse), though all other cases
are resolved. (Also, from casual examination, it appears that cargo
could in fact employ the proposed defaults without a problem, though
the types would be different than the types as they appear in the
source today, but this has not been fully verified.)

# Detailed design

The high-level rule is as stated: If the object type appears within
one or more reference types (`&` or `&mut`), then the default lifetime
bound is taken from the innermost enclosing reference. Otherwise the
default is `'static`.

**Precedence of this rule with respect to other defaults.** This rule
takes precedence over the existing existing defaults that are applied
in function signatures as well as those that are intended (but not yet
implemented) for `impl` declarations. Therefore:

```rust
fn foo1(obj: &SomeTrait) { }
fn foo2(obj: Box<SomeTrait>) { }
```

expand under this RFC to:

```rust
// Under this RFC:
fn foo1<'a>(obj: &'a (SomeTrait+'a)) { }
fn foo2(obj: Box<SomeTrait+'static>) { }
```

whereas today those same functions expand to:

```rust
// Under existing rules:
fn foo1<'a,'b>(obj: &'a (SomeTrait+'b)) { }
fn foo2(obj: Box<SomeTrait+'static>) { }
```

The reason for this rule is that we wish to ensure that if one writes
a struct declaration, then any types which appear in the struct
declaration can be safely copy-and-pasted into a fn signature. For example:

```rust
struct Foo {
    x: Box<SomeTrait>, // equiv to `Box<SomeTrait+'static>`
}

fn bar(foo: &mut Foo, x: Box<SomeTrait>) {
    foo.x = x; // (*)
}
```

The goal is to ensure that the line marked with `(*)` continues to
compile. If we gave the fn signature defaults precedence over the
object defaults, the assignment would in this case be illegal, because
the expansion of `Box<SomeTrait>` would be different.

**Interaction with object coercion.** The rules specify that `&'a
SomeTrait` and `&'a mut SomeTrait` are expanded to `&'a
(SomeTrait+'a)`and `&'a mut (SomeTrait+'a)` respecively. Today, in fn
signatures, one would get the expansions `&'a (SomeTrait+'b)` and `&'a
mut (SomeTrait+'b)`, respectively. In the case of a shared reference
`&'a SomeTrait`, this difference is basically irrelevant, as the
lifetime bound can always be approximated to be shorter when needed.

In the case a mutable reference `&'a mut SomeTrait`, however, using
two lifetime variables is *in principle* a more general expansion. The
reason has to do with "variance" -- specifically, because the proposed
expansion places the `'a` lifetime qualifier in the reference of a
mutable reference, the compiler will be unable to allow `'a` to be
approximated with a shorter lifetime. You may have experienced this if
you have types like `&'a mut &'a mut Foo`; the compiler is also forced
to be conservative about the lifetime `'a` in that scenario.

However, in the specific case of object types, this concern is
ameliorated by the existing object coercions. These coercions permit
`&'a mut (SomeTrait+'a)` to be coerced to `&'b mut (SomeTrait+'c)`
where `'a : 'b` and `'a : 'c`. The reason that this is legal is
because unsized types (like object types) cannot be assigned, thus
sidestepping the variance concerns. This means that programs like the
following compile successfully (though you will find that you get
errors if you replace the object type `(Counter+'a)` with the
underlying type `&'a mut u32`):

```rust
#![allow(unused_variables)]
#![allow(dead_code)]

trait Counter {
    fn inc_and_get(&mut self) -> u32;
}

impl<'a> Counter for &'a mut u32 {
    fn inc_and_get(&mut self) -> u32 {
        **self += 1;
        **self
    }
}

fn foo<'a>(x: &'a u32, y: &'a mut (Counter+'a)) {
}

fn bar<'a>(x: &'a mut (Counter+'a)) {
    let value = 2_u32;
    foo(&value, x)
}

fn main() {
}
```

This may seem surprising, but it's a reflection of the fact that
object types give the user less power than if the user had direct
access to the underlying data; the user is confined to accessing the
underlying data through a known interface.

# Drawbacks

**A. Breaking change.** This change has the potential to break some
existing code, though given the statistics gathered we believe the
effect will be minimal (in particular, defaults are only permitted in
fn signatures today, so in most existing code explicit lifetime bounds
are used).

**B. Hidden references get the wrong default.** If you have a struct
with a reference hidden within one of the fields, the defaults that
result might not be what you desire. Consider the following:

```rust
struct Ref<'a, T: ?Sized + 'a> {
    reference: &'a T
}
```

Now if I have a type `Ref<'a, SomeTrait>`, I might prefer for this
type to expand to `Ref<'a, SomeTrait+'a>`. However, given the rules as
proposed here, the result would be `Ref<'a, SomeTrait+'static>`. If
this becomes a serious annoyance, in the future we could add the
option for declaring on the struct `Ref` that objects which appear in
type `T` should be defaulted with the lifetime `'a`.

**C. Lifetime errors with defaults can get confusing.** Defaults
always carry some potential to surprise users, though it's worth
pointing out that the current rules are also a big source of
confusion. Further improvements like the current system for suggesting
alternative fn signatures would help here, of course (and are an
expected subject of investigation regardless).

# Alternatives

1. **Leave things as they are with an improved error message.**
Besides the general dissatisfaction with the current system, a big
concern here is that if [RFC 458][458] is accepted (which seems
likely), this implies that object types like `SomeTrait+Send` will now
require an explicit region bound. Most of the time, that would be
`SomeTrait+Send+'static`, which is very long indeed. We considered the
option of introducing a new trait, let's call it `Own` for now, that
is basically `Send+'static`. However, that required (1) finding a
reasonable name for `Own`; (2) seems to lessen one of the benefits of
[RFC 458][458], which is that lifetimes and other properties can be
considered orthogonally; and (3) does nothing to help with cases like
`&'a mut FnMut()`, which one would still have to write as `&'a mut
(FnMut()+'a)`.

2. **Drive defaults with the `T:'a` annotations that appear on
structs.** To address Drawback B, the fact that hidden references get
the wrong default, we briefly considered driving defaults on arbitrary
structs using the `T:'a` annotations that are (currently) required.
However, this carries some downsides: (1) there can be more than one
such annotation, raising the risk of ambiguity; (2) in the (perhaps
not so distant) future we would like to introduce a plan to infer
those annotations, which is incompatible with defaults; (3) the
defaulting is then more complicated and perhaps more confusing
overall. Finally, it is not clear whether Drawback B will be a real
problem in practice.

# Unresolved questions

None.

[34]: https://github.com/rust-lang/rfcs/blob/master/text/0034-bounded-type-parameters.md
[16948]: https://github.com/rust-lang/rust/issues/16948
[458]: https://github.com/rust-lang/rfcs/pull/458
