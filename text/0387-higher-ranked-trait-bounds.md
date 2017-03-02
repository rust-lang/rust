- Start Date: 2014-10-10
- RFC PR: [rust-lang/rfcs#387](https://github.com/rust-lang/rfcs/pull/387)
- Rust Issue: [rust-lang/rust#18639](https://github.com/rust-lang/rust/issues/18639)

# Summary

- Add the ability to have trait bounds that are polymorphic over lifetimes.

# Motivation

Currently, closure types can be polymorphic over lifetimes. But
closure types are deprecated in favor of traits and object types as
part of RFC #44 (unboxed closures). We need to close the gap. The
canonical example of where you want this is if you would like a
closure that accepts a reference with any lifetime. For example,
today you might write:

```rust
fn with(callback: |&Data|) {
    let data = Data { ... };
    callback(&data)
}
```

If we try to write this using unboxed closures today, we have a problem:

```
fn with<'a, T>(callback: T)
    where T : FnMut(&'a Data)
{
    let data = Data { ... };
    callback(&data)
}

// Note that the `()` syntax is shorthand for the following:
fn with<'a, T>(callback: T)
    where T : FnMut<(&'a Data,),()>
{
    let data = Data { ... };
    callback(&data)
}
```
    
The problem is that the argument type `&'a Data` must include a
lifetime, and there is no lifetime one could write in the fn sig that
represents "the stack frame of the `with` function". Naturally
we have the same problem if we try to use an `FnMut` object (which is
the closer analog to the original closure example):

```rust
fn with<'a>(callback: &mut FnMut(&'a Data))
{
    let data = Data { ... };
    callback(&data)
}

fn with<'a>(callback: &mut FnMut<(&'a Data,),()>)
{
    let data = Data { ... };
    callback(&data)
}
```

Under this proposal, you would be able to write this code as follows:

```
// Using the FnMut(&Data) notation, the &Data is
// in fact referencing an implicit bound lifetime, just
// as with closures today.
fn with<T>(callback: T)
    where T : FnMut(&Data)
{
    let data = Data { ... };
    callback(&data)
}

// If you prefer, you can use an explicit name,
// introduced by the `for<'a>` syntax.
fn with<T>(callback: T)
    where T : for<'a> FnMut(&'a Data)
{
    let data = Data { ... };
    callback(&data)
}

// No sugar at all.
fn with<T>(callback: T)
    where T : for<'a> FnMut<(&'a Data,),()>
{
    let data = Data { ... };
    callback(&data)
}
```
    
And naturally the object form(s) work as well:    

```rust
// The preferred notation, using `()`, again introduces
// implicit binders for omitted lifetimes:
fn with(callback: &mut FnMut(&Data))
{
    let data = Data { ... };
    callback(&data)
}

// Explicit names work too.
fn with(callback: &mut for<'a> FnMut(&'a Data))
{
    let data = Data { ... };
    callback(&data)
}

// The fully explicit notation requires an explicit `for`,
// as before, to declare the bound lifetimes.
fn with(callback: &mut for<'a> FnMut<(&'a Data,),()>)
{
    let data = Data { ... };
    callback(&data)
}
```

The syntax for `fn` types must be updated as well to use `for`.

# Detailed design

## For syntax

We modify the grammar for a trait reference to include

    for<lifetimes> Trait<T1, ..., Tn>
    for<lifetimes> Trait(T1, ..., tn) -> Tr

This syntax can be used in where clauses and types. The `for` syntax
is not permitted in impls nor in qualified paths (`<T as Trait>`).  In
impls, the distinction between early and late-bound lifetimes are
inferred. In qualified paths, which are used to select a member from
an impl, no bound lifetimes are permitted.

## Update syntax of fn types

The existing bare fn types will be updated to use the same `for`
notation. Therefore, `<'a> fn(&'a int)` becomes `for<'a> fn(&'a int)`.

## Implicit binders when using parentheses notation and in fn types

When using the `Trait(T1, ..., Tn)` notation, implicit binders are
introduced for omitted lifetimes. In other words, `FnMut(&int)` is
effectively shorthand for `for<'a> FnMut(&'a int)`, which is itself
shorthand for `for<'a> FnMut<(&'a int,),()>`. No implicit binders are
introduced when not using the parentheses notation (i.e.,
`Trait<T1,...,Tn>`). These binders interact with lifetime elision in
the usual way, and hence `FnMut(&Foo) -> &Bar` is shorthand for
`for<'a> FnMut(&'a Foo) -> &'a Bar`. The same is all true (and already
true) for fn types.

## Distinguishing early vs late bound lifetimes in impls

We will distinguish early vs late-bound lifetimes on impls in the same
way as we do for fns. Background on this process can be found in these
two blog posts \[[1][1], [2][2]\]. The basic idea is to distinguish
early-bound lifetimes, which must be substituted immediately, from
late-bound lifetimes, which can be made into a higher-ranked trait
reference.

The rule is that any lifetime parameter `'x` declared on an impl is
considered *early bound* if `'x` appears in any of the following locations:

- the self type of the impl;
- a where clause associated with the impl (here we assume that all bounds on
  impl parameters are desugared into where clauses). 
 
All other lifetimes are considered *late bound*.

When we decide what kind of trait-reference is *provided* by an impl,
late bound lifetimes are moved into a `for` clause attached to the
reference. Here are some examples:

```rust
// Here 'late does not appear in any where clause nor in the self type,
// and hence it is late-bound. Thus this impl is considered to provide:
//
//     SomeType : for<'late> FnMut<(&'late Foo,),()>
impl<'late> FnMut(&'late Foo) -> Bar for SomeType { ... }

// Here 'early appears in the self type and hence it is early bound.
// This impl thus provides:
//
//     SomeOtherType<'early> : FnMut<(&'early Foo,),()>
impl<'early> FnMut(&'early Foo) -> Bar for SomeOtherType<'early> { ... }
```

This means that if there were a consumer that required a type which
implemented `FnMut(&Foo)`, only `SomeType` could be used, not
`SomeOtherType`:

```rust
fn foo<T>(t: T) where T : FnMut(&Foo) { ... }

foo::<SomeType>(...) // ok
foo::<SomeOtherType<'static>>(...) // not ok
```

[1]: http://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
[2]: http://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/

## Instantiating late-bound lifetimes in a trait reference

Whenever
an associated item from a trait reference is accessed, all late-bound
lifetimes are instantiated. This means basically when a method is
called and so forth.  Here are some examples:

    fn foo<'b,T:for<'a> FnMut(&'a &'b Foo)>(t: T) {
        t(...); // here, 'a is freshly instantiated
        t(...); // here, 'a is freshly instantiated again
    }
    
Other times when a late-bound lifetime would be instantiated:

- Accessing an associated constant, once those are implemented.
- Accessing an associated type.

Another way to state these rules is that bound lifetimes are not
permitted in the traits found in qualified paths -- and things like
method calls and accesses to associated items can all be desugared
into calls via qualified paths. For example, the call `t(...)` above
is equivalent to:

    fn foo<'b,T:for<'a> FnMut(&'a &'b Foo)>(t: T) {
        // Here, per the usual rules, the omitted lifetime on the outer
        // reference will be instantiated with a fresh variable.
        <t as FnMut<(&&'b Foo,),()>::call_mut(&mut t, ...);
        <t as FnMut<(&&'b Foo,),()>::call_mut(&mut t, ...);
    }
    
## Subtyping of trait references

The subtyping rules for trait references that involve higher-ranked
lifetimes will be defined in an analogous way to the current subtyping
rules for closures. The high-level idea is to replace each
higher-ranked lifetime with a skolemized variable, perform the usual
subtyping checks, and then check whether those skolemized variables
would be being unified with anything else. The interested reader is
referred to
[Simon Peyton-Jones rather thorough but quite readable paper on the topic][spj]
or the documentation in
`src/librustc/middle/typeck/infer/region_inference/doc.rs`.

The most important point is that the rules provide for subtyping that
goes from "more general" to "less general". For example, if I have a
trait reference like `for<'a> FnMut(&'a int)`, that would be usable
wherever a trait reference with a concrete lifetime, like
`FnMut(&'static int)`, is expected.
   
[spj]: http://research.microsoft.com/en-us/um/people/simonpj/papers/higher-rank/

# Drawbacks

This feature is needed. There isn't really any particular drawback beyond
language complexity.

# Alternatives

**Drop the keyword.** The `for` keyword is used due to potential
ambiguities surrounding UFCS notation. Under UFCS, it is legal to
write e.g. `<T>::Foo::Bar` in a type context. This is awfully close to
something like `<'a> ::std::FnMut`. Currently, the parser could
probably use the lifetime distinction to know the difference, but
future extensions (see next paragraph) could allow types to be used as
well, and it is still possible we will opt to "drop the tick" in
lifetimes. Moreover, the syntax `<'a> FnMut(&'a uint)` is not exactly
beautiful to begin with.

**Permit higher-ranked traits with type variables.** This RFC limits
"higher-rankedness" to lifetimes. It is plausible to extend the system
in the future to permit types as well, though only in where clauses
and not in types. For example, one might write:

    fn foo<IDENTITY>(t: IDENTITY) where IDENTITY : for<U> FnMut(U) -> U { ... }
    
# Unresolved questions

None. Implementation is underway though not complete.
