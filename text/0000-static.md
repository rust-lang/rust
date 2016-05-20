- Feature Name: static_lifetime_in_statics
- Start Date: 2016-05-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Let's default lifetimes in static and const declarations to `'static`.

# Motivation
[motivation]: #motivation

Currently, having references in `static` and `const` declarations is cumbersome 
due to having to explicitly write `&'static ..`. On the other hand anything but 
static is likely either useless, unsound or both. Also the long lifetime name 
causes substantial rightwards drift, which makes it hard to format the code 
to be visually appealing.

For example, having a `'static` default for lifetimes would turn this:
```
static my_awesome_tables: &'static [&'static HashMap<Cow<'static, str>, u32>] = ..
```
into this:
```
static my_awesome_table: &[&HashMap<Cow<str>, u32>] = ..
```

The type declaration still causes some rightwards drift, but at least all the
contained information is useful.

# Detailed design
[design]: #detailed-design

The same default that RFC #599 sets up for trait object is to be used for 
statics and const declarations. In those declarations, the compiler will assume 
`'static` when a lifetime is not explicitly given in both refs and generics.

Note that this RFC does not forbid writing the lifetimes, it only sets a 
default when no is given. Thus the change is unlikely to cause any breakage and 
should be deemed backwards-compatible. It's also very unlikely that 
implementing this RFC will restrict our design space for `static` and `const` 
definitions down the road.

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
* Infer types for statics. The absence of types makes it harder to reason about
the code, so even if type inference for statics was to be implemented, 
defaulting lifetimes would have the benefit of pulling the cost-benefit 
relation in the direction of more explicit code. Thus it is advisable to 
implement this change even with the possibility of implementing type inference 
later.

# Unresolved questions
[unresolved]: #unresolved-questions

* Does this change requires changing the grammar?
* Are there other Rust-code handling programs that need to be updated?
