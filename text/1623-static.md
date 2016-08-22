- Feature Name: static_lifetime_in_statics
- Start Date: 2016-05-20
- RFC PR: https://github.com/rust-lang/rfcs/pull/1623
- Rust Issue: https://github.com/rust-lang/rust/issues/35897

# Summary
[summary]: #summary

Let's default lifetimes in static and const declarations to `'static`.

# Motivation
[motivation]: #motivation

Currently, having references in `static` and `const` declarations is cumbersome 
due to having to explicitly write `&'static ..`. Also the long lifetime name 
causes substantial rightwards drift, which makes it hard to format the code 
to be visually appealing.

For example, having a `'static` default for lifetimes would turn this:
```rust
static my_awesome_tables: &'static [&'static HashMap<Cow<'static, str>, u32>] = ..
```
into this:
```rust
static my_awesome_table: &[&HashMap<Cow<str>, u32>] = ..
```

The type declaration still causes some rightwards drift, but at least all the
contained information is useful. There is one exception to the rule: lifetime
elision for function signatures will work as it does now (see example below).

# Detailed design
[design]: #detailed-design

The same default that RFC #599 sets up for trait object is to be used for 
statics and const declarations. In those declarations, the compiler will assume 
`'static` when a lifetime is not explicitly given in all reference lifetimes,
including reference lifetimes obtained via generic substitution.

Note that this RFC does not forbid writing the lifetimes, it only sets a 
default when no is given. Thus the change will not cause any breakage and is 
therefore backwards-compatible. It's also very unlikely that implementing this 
RFC will restrict our design space for `static` and `const` definitions down 
the road.

The `'static` default does *not* override lifetime elision in function 
signatures, but work alongside it:

```rust
static foo: fn(&u32) -> &u32 = ...;  // for<'a> fn(&'a u32) -> &'a u32
static bar: &Fn(&u32) -> &u32 = ...; // &'static for<'a> Fn(&'a u32) -> &'a u32
```

With generics, it will work as anywhere else, also differentiating between
function lifetimes and reference lifetimes. Notably, writing out the lifetime
is still possible.

```rust
trait SomeObject<'a> { .. }
static foo: &SomeObject = ...; // &'static SomeObject<'static>
static bar: &for<'a> SomeObject<'a> = ...; // &'static for<'a> SomeObject<'a>
static baz: &'static [u8] = ...;

struct SomeStruct<'a, 'b> {
    foo: &'a Foo,
    bar: &'a Bar,
    f: for<'b> Fn(&'b Foo) -> &'b Bar
}

static blub: &SomeStruct = ...; // &'static SomeStruct<'static, 'b> for any 'b
```

It will still be an error to omit lifetimes in function types *not* eligible 
for elision, e.g.

```rust
static blobb: FnMut(&Foo, &Bar) -> &Baz = ...; //~ ERROR: missing lifetimes for
                                               //^ &Foo, &Bar, &Baz
```

This ensures that the really hairy cases that need the full type documented
aren't unduly abbreviated.

It should also be noted that since statics and constants have no `self` type,
elision will only work with distinct input lifetimes or one input+output
lifetime.

# Drawbacks
[drawbacks]: #drawbacks

There are no known drawbacks to this change.

# Alternatives
[alternatives]: #alternatives

* Leave everything as it is. Everyone using static references is annoyed by 
having to add `'static` without any value to readability. People will resort to 
writing macros if they have many resources.
* Write the aforementioned macro. This is inferior in terms of UX. Depending on
the implementation it may or may not be possible to default lifetimes in
generics.
* Make all non-elided lifetimes `'static`. This has the drawback of creating
hard-to-spot errors (that would also probably occur in the wrong place) and
confusing users.
* Make all non-declared lifetimes `'static`. This would not be backwards
compatible due to interference with lifetime elision.
* Infer types for statics. The absence of types makes it harder to reason about
the code, so even if type inference for statics was to be implemented, 
defaulting lifetimes would have the benefit of pulling the cost-benefit 
relation in the direction of more explicit code. Thus it is advisable to 
implement this change even with the possibility of implementing type inference 
later.

# Unresolved questions
[unresolved]: #unresolved-questions

* Are there third party Rust-code handling programs that need to be updated to
deal with this change?
