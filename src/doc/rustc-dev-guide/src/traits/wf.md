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

Well-formedness checking happens in the [`chalk/chalk-rules/src/wf.rs`][wf]
module in chalk. After you have read this chapter, you may find useful to see
an extended set of examples in the [`chalk/src/test/wf.rs`][wf_test] submodule.

The new-style WF checking has not been implemented in rustc yet.

[wf]: https://github.com/rust-lang/chalk/blob/master/chalk-rules/src/wf.rs
[wf_test]: https://github.com/rust-lang/chalk/blob/master/src/test/wf.rs

We give here a complete reference of the generated goals for each Rust
declaration.

In addition to the notations introduced in the chapter about
lowering rules, we'll introduce another notation: when checking WF of a
declaration, we'll often have to prove that all types that appear are
well-formed, except type parameters that we always assume to be WF. Hence,
we'll use the following notation: for a type `SomeType<...>`, we define
`InputTypes(SomeType<...>)` to be the set of all non-parameter types appearing
in `SomeType<...>`, including `SomeType<...>` itself.

Examples:
* `InputTypes((u32, f32)) = [u32, f32, (u32, f32)]`
* `InputTypes(Box<T>) = [Box<T>]` (assuming that `T` is a type parameter)
* `InputTypes(Box<Box<T>>) = [Box<T>, Box<Box<T>>]`

We also extend the `InputTypes` notation to where clauses in the natural way.
So, for example `InputTypes(A0: Trait<A1,...,An>)` is the union of
`InputTypes(A0)`, `InputTypes(A1)`, ..., `InputTypes(An)`.

# Type definitions

Given a general type definition:
```rust,ignore
struct Type<P...> where WC_type {
    field1: A1,
    ...
    fieldn: An,
}
```

we generate the following goal, which represents its well-formedness condition:
```text
forall<P...> {
    if (FromEnv(WC_type)) {
        WellFormed(InputTypes(WC_type)) &&
            WellFormed(InputTypes(A1)) &&
            ...
            WellFormed(InputTypes(An))
    }
}
```

which in English states: assuming that the where clauses defined on the type
hold, prove that every type appearing in the type definition is well-formed.

Some examples:
```rust,ignore
struct OnlyClone<T> where T: Clone {
    clonable: T,
}
// The only types appearing are type parameters: we have nothing to check,
// the type definition is well-formed.
```

```rust,ignore
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
```

```rust,ignore
struct Bar<T> where <T as Iterator>::Item: Debug {
    bar: i32,
}
// The only non-parameter types which appear in this definition are
// `<T as Iterator>::Item` and `i32`. The generated goal is the following:
// ```
// forall<T> {
//     if (FromEnv(<T as Iterator>::Item: Debug)) {
//          WellFormed(<T as Iterator>::Item) &&
//               WellFormed(i32)
//     }
// }
// ```
// which is not provable since `WellFormed(<T as Iterator>::Item)` requires
// proving `Implemented(T: Iterator)`, and we are unable to prove that for an
// unknown `T`.
//
// Hence, this type definition is considered illegal. An additional
// `where T: Iterator` would make it legal.
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
trait Foo<T> where T: Iterator, <T as Iterator>::Item: Debug {
    ...
}
// The only non-parameter type which appears in this definition is
// `<T as Iterator>::Item`. The generated goal is the following:
// ```
// forall<T> {
//     if (FromEnv(T: Iterator), FromEnv(<T as Iterator>::Item: Debug)) {
//         WellFormed(<T as Iterator>::Item)
//     }
// }
// ```
// which is provable thanks to the `FromEnv(T: Iterator)` assumption.
```

```rust,ignore
trait Bar {
    type Assoc<T>: From<<T as Iterator>::Item>;
}
// The only non-parameter type which appears in this definition is
// `<T as Iterator>::Item`. The generated goal is the following:
// ```
// forall<T> {
//     WellFormed(<T as Iterator>::Item)
// }
// ```
// which is not provable, hence the trait definition is considered illegal.
```

```rust,ignore
trait Baz {
    type Assoc<T>: From<<T as Iterator>::Item> where T: Iterator;
}
// The generated goal is now:
// ```
// forall<T> {
//     if (FromEnv(T: Iterator)) {
//         WellFormed(<T as Iterator>::Item)
//     }
// }
// ```
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

impl<U> Foo<Option<U>> for () {
    // We substitute type parameters from the trait by the ones provided
    // by the impl, that is instead of having a `T: Clone` where clause,
    // we have an `Option<U>: Clone` one.
    type Assoc = OnlyClone<Option<U>> where Option<U>: Clone;
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

> So in Rust, where clauses on associated types work *exactly* like where
> clauses on trait methods: in an impl, we must substitute the parameters from
> the traits with values provided by the impl, we may omit them if we don't
> need them, but we cannot add new where clauses.

Now let's see the generated goal for this general impl:
```text
forall<P1...> {
    // Well-formedness of types appearing in the impl
    if (FromEnv(WC_impl), FromEnv(InputTypes(SomeType<A2...>: Trait<A1...>))) {
        WellFormed(InputTypes(WC_impl)) &&

            forall<P2...> {
                if (FromEnv(WC_assoc)) {
                        WellFormed(InputTypes(SomeValue<A3...>))
                }
            }
    }

    // Implied bounds checking
    if (FromEnv(WC_impl), FromEnv(InputTypes(SomeType<A2...>: Trait<A1...>))) {
        WellFormed(SomeType<A2...>: Trait<A1...>) &&

            forall<P2...> {
                if (FromEnv(WC_assoc)) {
                    WellFormed(SomeValue<A3...>: Bounds_assoc)
                }
            }
    }
}
```

Here is the most complex goal. As always, first, assuming that
the various where clauses hold, we prove that every type appearing in the impl
is well-formed, ***except*** types appearing in the impl header
`SomeType<A2...>: Trait<A1...>`. Instead, we *assume* that those types are
well-formed
(hence the `if (FromEnv(InputTypes(SomeType<A2...>: Trait<A1...>)))`
conditions). This is
part of the implied bounds proposal, so that we can rely on the bounds
written on the definition of e.g. the `SomeType<A2...>` type (and that we don't
need to repeat those bounds).
> Note that we don't need to check well-formedness of types appearing in
> `WC_assoc` because we already did that in the trait decl (they are just
> repeated with some substitutions of values which we already assume to be
> well-formed)

Next, still assuming that the where clauses on the impl `WC_impl` hold and that
the input types of `SomeType<A2...>` are well-formed, we prove that
`WellFormed(SomeType<A2...>: Trait<A1...>)` hold. That is, we want to prove
that `SomeType<A2...>` verify all the where clauses that might transitively
be required by the `Trait` definition (see
[this subsection](./implied-bounds.md#co-inductiveness-of-wellformed)).

Lastly, assuming in addition that the where clauses on the associated type
`WC_assoc` hold,
we prove that `WellFormed(SomeValue<A3...>: Bounds_assoc)` hold. Again, we are
not only proving `Implemented(SomeValue<A3...>: Bounds_assoc)`, but also
all the facts that might transitively come from `Bounds_assoc`. We must do this
because we allow the use of implied bounds on associated types: if we have
`FromEnv(SomeType: Trait)` in our environment, the lowering rules
chapter indicates that we are able to deduce
`FromEnv(<SomeType as Trait>::Assoc: Bounds_assoc)` without knowing what the
precise value of `<SomeType as Trait>::Assoc` is.

Some examples for the generated goal:
```rust,ignore
// Trait Program Clauses

// These are program clauses that come from the trait definitions below
// and that the trait solver can use for its reasonings. I'm just restating
// them here so that we have them in mind.

trait Copy { }
// This is a program clause that comes from the trait definition above
// and that the trait solver can use for its reasonings. I'm just restating
// it here (and also the few other ones coming just after) so that we have
// them in mind.
// `WellFormed(Self: Copy) :- Implemented(Self: Copy).`

trait Partial where Self: Copy { }
// ```
// WellFormed(Self: Partial) :-
//     Implemented(Self: Partial) &&
//     WellFormed(Self: Copy).
// ```

trait Complete where Self: Partial { }
// ```
// WellFormed(Self: Complete) :-
//     Implemented(Self: Complete) &&
//     WellFormed(Self: Partial).
// ```

// Impl WF Goals

impl<T> Partial for T where T: Complete { }
// The generated goal is:
// ```
// forall<T> {
//     if (FromEnv(T: Complete)) {
//         WellFormed(T: Partial)
//     }
// }
// ```
// Then proving `WellFormed(T: Partial)` amounts to proving
// `Implemented(T: Partial)` and `Implemented(T: Copy)`.
// Both those facts can be deduced from the `FromEnv(T: Complete)` in our
// environment: this impl is legal.

impl<T> Complete for T { }
// The generated goal is:
// ```
// forall<T> {
//     WellFormed(T: Complete)
// }
// ```
// Then proving `WellFormed(T: Complete)` amounts to proving
// `Implemented(T: Complete)`, `Implemented(T: Partial)` and
// `Implemented(T: Copy)`.
//
// `Implemented(T: Complete)` can be proved thanks to the
// `impl<T> Complete for T` blanket impl.
//
// `Implemented(T: Partial)` can be proved thanks to the
// `impl<T> Partial for T where T: Complete` impl and because we know
// `T: Complete` holds.

// However, `Implemented(T: Copy)` cannot be proved: the impl is illegal.
// An additional `where T: Copy` bound would be sufficient to make that impl
// legal.
```

```rust,ignore
trait Bar { }

impl<T> Bar for T where <T as Iterator>::Item: Bar { }
// We have a non-parameter type appearing in the where clauses:
// `<T as Iterator>::Item`. The generated goal is:
// ```
// forall<T> {
//     if (FromEnv(<T as Iterator>::Item: Bar)) {
//         WellFormed(T: Bar) &&
//             WellFormed(<T as Iterator>::Item: Bar)
//     }
// }
// ```
// And `WellFormed(<T as Iterator>::Item: Bar)` is not provable: we'd need
// an additional `where T: Iterator` for example.
```

```rust,ignore
trait Foo { }

trait Bar {
    type Item: Foo;
}

struct Stuff<T> { }

impl<T> Bar for Stuff<T> where T: Foo {
    type Item = T;
}
// The generated goal is:
// ```
// forall<T> {
//     if (FromEnv(T: Foo)) {
//         WellFormed(T: Foo).
//     }
// }
// ```
// which is provable.
```

```rust,ignore
trait Debug { ... }
// `WellFormed(Self: Debug) :- Implemented(Self: Debug).`

struct Box<T> { ... }
impl<T> Debug for Box<T> where T: Debug { ... }

trait PointerFamily {
    type Pointer<T>: Debug where T: Debug;
}
// `WellFormed(Self: PointerFamily) :- Implemented(Self: PointerFamily).`

struct BoxFamily;

impl PointerFamily for BoxFamily {
    type Pointer<T> = Box<T> where T: Debug;
}
// The generated goal is:
// ```
// forall<T> {
//     WellFormed(BoxFamily: PointerFamily) &&
//
//     if (FromEnv(T: Debug)) {
//         WellFormed(Box<T>: Debug) &&
//             WellFormed(Box<T>)
//     }
// }
// ```
// `WellFormed(BoxFamily: PointerFamily)` amounts to proving
// `Implemented(BoxFamily: PointerFamily)`, which is ok thanks to our impl.
//
// `WellFormed(Box<T>)` is always true (there are no where clauses on the
// `Box` type definition).
//
// Moreover, we have an `impl<T: Debug> Debug for Box<T>`, hence
// we can prove `WellFormed(Box<T>: Debug)` and the impl is indeed legal.
```

```rust,ignore
trait Foo {
    type Assoc<T>;
}

struct OnlyClone<T: Clone> { ... }

impl Foo for i32 {
    type Assoc<T> = OnlyClone<T>;
}
// The generated goal is:
// ```
// forall<T> {
//     WellFormed(i32: Foo) &&
//        WellFormed(OnlyClone<T>)
// }
// ```
// however `WellFormed(OnlyClone<T>)` is not provable because it requires
// `Implemented(T: Clone)`. It would be tempting to just add a `where T: Clone`
// bound inside the `impl Foo for i32` block, however we saw that it was
// illegal to add where clauses that didn't come from the trait definition.
```
