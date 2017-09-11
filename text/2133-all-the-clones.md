- Feature Name: `all_the_clones`
- Start Date: 2017-08-28
- RFC PR: https://github.com/rust-lang/rfcs/pull/2133
- Rust Issue: https://github.com/rust-lang/rust/issues/44496

# Summary
[summary]: #summary

Add compiler-generated `Clone` implementations for tuples and arrays with `Clone` elements of all lengths.

# Motivation
[motivation]: #motivation

Currently, the `Clone` trait for arrays and tuples is implemented using a [macro] in libcore, for tuples of size 11 or less and for `Copy` arrays of size 32 or less. This breaks the uniformity of the language and annoys users.

Also, the compiler already implements `Copy` for all arrays and tuples with all elements `Copy`, which forces the compiler to provide an implementation for `Copy`'s supertrait `Clone`. There is no reason the compiler couldn't provide `Clone` impls for all arrays and tuples.

[macro]: https://github.com/rust-lang/rust/blob/f3d6973f41a7d1fb83029c9c0ceaf0f5d4fd7208/src/libcore/tuple.rs#L25

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Arrays and tuples of `Clone` arrays are `Clone` themselves. Cloning them clones all of their elements.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Make `clone` a lang-item, add the following trait rules to the compiler:

```
n number
T type
T: Clone
----------
[T; n]: Clone

T1,...,Tn types
T1: Clone, ..., Tn: Clone
----------
(T1, ..., Tn): Clone
```

And add the obvious implementations of `Clone::clone` and `Clone::clone_from` as MIR shim implementations, in the same manner as `drop_in_place`. The implementations could also do a shallow copy if the type ends up being `Copy`.

Remove the macro implementations in libcore. We still have macro implementations for other "derived" traits, such as `PartialEq`, `Hash`, etc.

Note that independently of this RFC, we're adding builtin `Clone` impls for all "scalar" types, most importantly fn pointer and fn item types (where manual impls are impossible in the foreseeable future because of higher-ranked types, e.g. `for<'a> fn(SomeLocalStruct<'a>)`), which are already `Copy`:
```
T fn pointer type
----------
T: Clone

T fn item type
----------
T: Clone

And just for completeness (these are perfectly done by an impl in Rust 1.19):

T int type | T uint type | T float type
----------
T: Clone

T type
----------
*const T: Clone
*mut T: Clone

T type
'a lifetime
----------
&'a T: Clone

----------
bool: Clone
char: Clone
!: Clone
```

This was considered a bug-fix (these types are all `Copy`, so it's easy to witness that they are `Clone`).

# Drawbacks
[drawbacks]: #drawbacks

The MIR shims add complexity to the compiler. Along with the `derive(Clone)` implementation in `libsyntax`, we have 2 separate sets of implementations of `Clone`. 

Having `Copy` and `Clone` impls for all arrays and tuples, but not `PartialEq` etc. impls, could be confusing to users.

# Rationale and Alternatives
[alternatives]: #alternatives

Even with all proposed expansions to Rust's type-system, for consistency, the compiler needs to have at least *some* built-in `Clone` implementations: the type `for<'a> fn(Foo<'a>)` is `Copy` for all user-defined types `Foo`, but there is no way to implement `Clone`, which is a supertrait of `Copy`, for it (an `impl<T> Clone for fn(T)` won't match against the higher-ranked type).

The MIR shims for `Clone` of arrays and tuples are actually pretty simple and don't add much complexity after we have `drop_in_place` and shims for `Copy` types.

## The array situation

In Rust 1.19, arrays are `Clone` only if they are `Copy`. This code does not compile:
```Rust
fn main() {
    let x = [Box::new(0)].clone(); //~ ERROR
    println!("{:?}", x[0]);
}
```

~~The reason (I think) is that there is no good way to write a variable-length array expression in macros. This wouldn't be fixed by the first iteration of const generics.~~ Actually, this can be done using a for-loop (`ArrayVec` is used here instead of a manual panic guard for simplicity, but it can be easily implemented given const generics).
```Rust
impl<const n: usize; T: Clone> Clone for [T; n] {
    fn clone(&self) -> Self {
        unsafe {
            let result : ArrayVec<Self> = ArrayVec::new();
            for elem in (self as &[T]) {
                result.push(elem.clone());
            }
            result.into_inner().unwrap()
        }
    }
}
```

OTOH, this means that making non-`Copy` arrays `Clone` is less of a bugfix and more of a new feature. It's however a nice feature - `[Box<u32>; 1]` not being `Clone` is an annoying and seemingly-pointless edge case.

## Implement `Clone` only for `Copy` types

As of Rust 1.19, the compiler *does not* have the `Clone` implementations, which causes ICEs such as [rust-lang/rust#25733] because `Clone` is a supertrait of `Copy`.

One alternative, which would solve ICEs while being conservative, would be to have compiler implementations for `Clone` only for *`Copy`* tuples of size 12+ and arrays, and maintain the `libcore` macros for `Clone` of tuples (in Rust 1.19, arrays are only `Clone` if they are `Copy`).

This would make the shims *trivial* (a `Clone` implementation for a `Copy` type is just a memcpy), and would not implement any features that are not needed.

When we get variadic generics, we could make all tuples with `Clone` elements `Clone`. When we get const generics, we could make all arrays with `Clone` elements `Clone`.

## Use a MIR implementation of `Clone` for all derived impls

The implementation on the other end of the conservative-radical end would be to use the MIR shims for *all* `#[derive(Clone)]` implementations. This would increase uniformity by getting rid of the separate `libsyntax` derived implementation. However:

1. We'll still need the `#[derive_Clone]` hook in libsyntax, which would presumably result in an attribute that trait selection can see. That's not a significant concern.

2. The more annoying issue is that, as a workaround to trait matching being inductive, derived implementations are imperfect - see [rust-lang/rust#26925]. This means that we either have to solve that issue for `Clone` (which is dedicatedly non-trivial) or have some sort of type-checking for the generated MIR shims, both annoying options.

3. A MIR shim implementation would also have to deal with edge cases such as `#[repr(packed)]`, which normal type-checking would handle for ordinary `derive`. I think drop glue already encounters all of these edge cases so we have to deal with them anyway.

## `Copy` and `Clone` for closures

We could also add implementations of `Copy` and `Clone` to closures. That is [RFC #2132] and should be discussed there.

# Unresolved questions
[unresolved]: #unresolved-questions

See Alternatives.

[RFC #2132]: https://github.com/rust-lang/rfcs/pull/2132
[rust-lang/rust#25733]: https://github.com/rust-lang/rust/issues/25733
[rust-lang/rust#26925]: https://github.com/rust-lang/rust/issues/26925
