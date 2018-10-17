# Well-formedness checking

WF checking has the job of checking that the various declarations in a Rust
program are well-formed. This is the basis for implied bounds, and partly for
that reason, this checking can be surprisingly subtle! For example, we
have to be sure that each impl proves the WF conditions declared on
the trait.

For each declaration in a Rust program, we will generate a logical goal and try
to prove it using the lowered rules we described in the
[lowering rules](./lowering-rules.md) chapter. If we are able to prove it, we
say that the construct is well-formed. If not, we report an error to the user.

Well-formedness checking happens in the [`src/rules/wf.rs`][wf] module in
chalk. After you have read this chapter, you may find useful to see an
extended set of examples in the [`src/rules/wf/test.rs`][wf_test] submodule.

The new-style WF checking has not been implemented in rustc yet.

[wf]: https://github.com/rust-lang-nursery/chalk/blob/master/src/rules/wf.rs
[wf_test]: https://github.com/rust-lang-nursery/chalk/blob/master/src/rules/wf/test.rs

We give here a complete reference of the generated goals for each Rust
declaration.

In addition with the notations introduced in the chapter about
lowering rules, we'll introduce another notation: when WF checking a
declaration, we'll often have to prove that all types that appear are
well-formed, except type parameters that we always assume to be WF. Hence,
we'll use the following notation: for a type `SomeType<...>`, we denote
`InputTypes(SomeType<...>)` the set of all non-parameter types appearing in
`SomeType<...>`, including `SomeType<...>` itself.

Examples:
* `InputTypes((u32, f32)) = [u32, f32, (u32, f32)]`
* `InputTypes(Box<T>) = [Box<T>]`
* `InputTypes(Box<Box<T>>) = [Box<T>, Box<Box<T>>]`

We may naturally extend the `InputTypes` notation to where clauses, for example
`InputTypes(A0: Trait<A1,...,An>)` is the union of `InputTypes(A0)`,
`InputTypes(A1)`, ..., `InputTypes(An)`.

# Type definitions

Given a general type definition:
```rust,ignore
struct Type<P...> where WC_type {
    field1: A1,
    ...
    fieldn: An,
}
```

we generate the following goal:
```
forall<P...> {
    if (FromEnv(WC_type)) {
        WellFormed(InputTypes(WC_type)) &&
            WellFormed(InputTypes(A1)) &&
            ...
            WellFormed(InputTypes(An))
    }
}
```

which in English gives: assuming that the where clauses defined on the type
hold, prove that every type appearing in the type definition is well-formed.

Some examples:
```rust,ignore
struct OnlyClone<T> where T: Clone {
    clonable: T,
}
// The only types appearing are type parameters: we have nothing to check,
// the type definition is well-formed.

struct Foo<T> where T: Clone {
    foo: OnlyClone<T>,
}
// The only non-parameter type which appears in this definition is
// `OnlyClone<T>`. The generated goal is the following:
// ```
// forall<T> {
//     if (FromEnv(T: Clone)) {
//          WellFormed(OnlyClone<T>)
//     }
// }
// ```
// which is provable.

struct Bar<T> where OnlyClone<T>: Debug {
    bar: i32,
}
// The only non-parameter type which appears in this definition is
// `OnlyClone<T>`. The generated goal is the following:
// ```
// forall<T> {
//     if (FromEnv(OnlyClone<T>: Debug)) {
//          WellFormed(OnlyClone<T>)
//     }
// }
// ```
// which is not provable since `WellFormed(OnlyClone<T>)` requires proving
// `Implemented(T: Clone)`, and we are unable to prove that for an unknown `T`.
// Hence, this type definition is considered illegal. An additional
// `where T: Clone` would make it legal.
```

# Trait definitions

Given a general trait definition:
```rust,ignore
trait Trait<P1...> where WC_trait {
    type Assoc<P2...>: Bounds_assoc where WC_assoc;
}
```

we generate the following goal:
```text
forall<P1...> {
    if (FromEnv(WC_trait)) {
        WellFormed(InputTypes(WC_trait)) &&

            forall<P2...> {
                if (FromEnv(WC_assoc)) {
                    WellFormed(InputTypes(Bounds_assoc)) &&
                        WellFormed(InputTypes(WC_assoc))
                }
            }
    }
}
```

There is not much to verify in a trait definition. We just want
to prove that the types appearing in the trait definition are well-formed,
under the assumption that the different where clauses hold.

Some examples:
```rust,ignore
struct OnlyClone<T: Clone> { ... }

trait Foo<T> where T: Clone, OnlyClone<T>: Debug {
    ...
}
// The only non-parameter type which appears in this definition is
// `OnlyClone<T>`. The generated goal is the following:
// ```
// forall<T> {
//     if (FromEnv(T: Clone), FromEnv(OnlyClone<T>: Debug)) {
//         WellFormed(OnlyClone<T>)
//     }
// }
// ```
// which is provable thanks to the `FromEnv(T: Clone)` assumption.

trait Bar {
    type Assoc<T>: From<OnlyClone<T>>;
}
// The only non-parameter type which appears in this definition is
// `OnlyClone<T>`. The generated goal is the following:
// forall<T> {
//     WellFormed(OnlyClone<T>)
// }
// which is not provable, hence the trait definition is considered illegal.

trait Baz {
    type Assoc<T>: From<OnlyClone<T>> where T: Clone;
}
// The generated goal is now:
// forall<T> {
//     if (FromEnv(T: Clone)) {
//         WellFormed(OnlyClone<T>)
//     }
// }
// which is now provable.
```

# Impls

Now we give ourselves a general impl for the trait defined above:
```rust,ignore
impl<P1...> Trait<A1...> for SomeType<A2...> where WC_impl {
    type Assoc<P2...> = SomeValue<A3...> where WC_assoc;
}
```

Note that here, `WC_assoc` are the same where clauses as those defined on the
associated type definition in the trait declaration, *except* that type
parameters from the trait are substituted with values provided by the impl
(see example below). You cannot add new where clauses. You may omit to write
the where clauses if you want to emphasize the fact that you are actually not
relying on them.

Some examples to illustrate that:
```rust,ignore
trait Foo<T> {
    type Assoc where T: Clone;
}

struct OnlyClone<T: Clone> { ... }

impl<T> Foo<Option<T>> for () {
    // We substitute type parameters from the trait by the ones provided
    // by the impl, that is instead of having a `T: Clone` where clause,
    // we have an `Option<T>: Clone` one.
    type Assoc = OnlyClone<Option<T>> where Option<T>: Clone;
}

impl<T> Foo<T> for i32 {
    // I'm not using the `T: Clone` where clause from the trait, so I can
    // omit it.
    type Assoc = u32;
}

impl<T> Foo<T> for f32 {
    type Assoc = OnlyClone<Option<T>> where Option<T>: Clone;
    //                                ^^^^^^^^^^^^^^^^^^^^^^
    //                                this where clause does not exist
    //                                on the original trait decl: illegal
}
```

So where clauses on associated types work *exactly* like where clauses on
trait methods: in an impl, we must substitute the parameters from the traits
with values provided by the impl, we may omit them if we don't need them, and
we cannot add new where clauses.

Now let's see the generated goal for this general impl:
```
forall<P1...> {
    if (FromEnv(WC_impl), FromEnv(InputTypes(SomeType<A2...>))) {
        WellFormed(SomeType<A2...>: Trait<A1...>) &&
            WellFormed(InputTypes(WC_impl)) &&

            forall<P2...> {
                if (FromEnv(WC_assoc)) {
                    WellFormed(SomeValue<A3...>: Bounds_assoc) &&
                        WellFormed(InputTypes(SomeValue<A3...>))
                }
            }
    }
}
```

Here is the most complex goal. As always, a first thing is that assuming that
the various where clauses hold, we prove that every type appearing in the impl
is well-formed, ***except*** types appearing in the receiver type
`SomeType<A2...>`. Instead, we *assume* that those types are well-formed
(hence the `if (FromEnv(InputTypes(SomeType<A2...>)))` condition). This is
part of the implied bounds proposal, so that we can rely on the bounds
written on the definition of the `SomeType<A2...>` type (and that we don't
need to repeat those bounds).

Next, assuming that the where clauses on the impl `WC_impl` hold and that the
input types of `SomeType<A2...>` are well-formed, we prove that
`WellFormed(SomeType<A2...>: Trait<A1...>)` hold. That is, we want to prove
that `SomeType<A2...>` verify all the where clauses that might transitively
come from the `Trait` definition (see
[this subsection](./implied-bounds#co-inductiveness-of-wellformed)).

Lastly, assuming that the where clauses on the associated type `WC_assoc` hold,
we prove that `WellFormed(SomeValue<A3...>: Bounds_assoc)` hold. Again, we are
not only proving `Implemented(SomeValue<A3...>: Bounds_assoc)`, but also
all the facts that might transitively come from `Bounds_assoc`. This is because
we allow the use of implied bounds on associated types: if we have
`FromEnv(SomeType: Trait)` in our environment, the lowering rules
chapter indicates that we are able to deduce
`FromEnv(<SomeType as Trait>::Assoc: Bounds_assoc)` without knowing what the
precise value of `<SomeType as Trait>::Assoc` is.
