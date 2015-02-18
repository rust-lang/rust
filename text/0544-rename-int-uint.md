- Start Date: 2014-12-28
- RFC PR #: [rust-lang/rfcs#544](https://github.com/rust-lang/rfcs/pull/544)
- Rust Issue #: [rust-lang/rust#20639](https://github.com/rust-lang/rust/issues/20639)

# Summary

This RFC proposes that we rename the pointer-sized integer types `int/uint`, so as to avoid misconceptions and misuses. After extensive community discussions and several revisions of this RFC, the finally chosen names are `isize/usize`.

# Motivation

Currently, Rust defines two [machine-dependent integer types](http://doc.rust-lang.org/reference.html#machine-dependent-integer-types) `int/uint` that have the same number of bits as the target platform's pointer type. These two types are used for many purposes: indices, counts, sizes, offsets, etc.

The problem is, `int/uint` *look* like default integer types, but pointer-sized integers are not good defaults, and it is desirable to discourage people from overusing them.

And it is a quite popular opinion that, the best way to discourage their use is to rename them.

Previously, the latest renaming attempt [RFC PR 464](https://github.com/rust-lang/rfcs/pull/464) was rejected. (Some parts of this RFC is based on that RFC.) [A tale of two's complement](http://discuss.rust-lang.org/t/a-tale-of-twos-complement/1062) states the following reasons:

- Changing the names would affect literally every Rust program ever written.
- Adjusting the guidelines and tutorial can be equally effective in helping people to select the correct type.
- All the suggested alternative names have serious drawbacks.

However:

Rust was and is undergoing quite a lot of breaking changes. Even though the `int/uint` renaming will "break the world", it is not unheard of, and it is mainly a "search & replace". Also, a transition period can be provided, during which `int/uint` can be deprecated, while the new names can take time to replace them. So "to avoid breaking the world" shouldn't stop the renaming.

`int/uint` have a long tradition of being the default integer type names, so programmers *will* be tempted to use them in Rust, even the experienced ones, no matter what the documentation says. The semantics of `int/uint` in Rust is quite different from that in many other mainstream languages. Worse, the Swift programming language, which is heavily influenced by Rust, has the types `Int/UInt` with *almost* the *same semantics* as Rust's `int/uint`, but it *actively encourages* programmers to use `Int` as much as possible. From [the Swift Programming Language](https://developer.apple.com/library/prerelease/ios/documentation/Swift/Conceptual/Swift_Programming_Language/TheBasics.html#//apple_ref/doc/uid/TP40014097-CH5-ID319):

> Swift provides an additional integer type, Int, which has the same size as the current platform’s native word size: ...

> Swift also provides an unsigned integer type, UInt, which has the same size as the current platform’s native word size: ...

> Unless you need to work with a specific size of integer, always use Int for integer values in your code. This aids code consistency and interoperability.

> Use UInt only when you specifically need an unsigned integer type with the same size as the platform’s native word size. If this is not the case, Int is preferred, even when the values to be stored are known to be non-negative.

Thus, it is very likely that newcomers will come to Rust, expecting `int/uint` to be the preferred integer types, *even if they know that they are pointer-sized*.

Not renaming `int/uint` violates the principle of least surprise, and is not newcomer friendly.

Before the rejection of [RFC PR 464](https://github.com/rust-lang/rfcs/pull/464), the community largely settled on two pairs of candidates: `imem/umem` and `iptr/uptr`. As stated in previous discussions, the names have some drawbacks that may be unbearable. (Please refer to [A tale of two's complement](http://discuss.rust-lang.org/t/a-tale-of-twos-complement/1062) and related discussions for details.)

This RFC originally proposed a new pair of alternatives `intx/uintx`.

However, given the discussions about the previous revisions of this RFC, and the discussions in [Restarting the `int/uint` Discussion]( http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131), this RFC author (@CloudiDust) now believes that `intx/uintx` are not ideal. Instead, one of the other pairs of alternatives should be chosen. The finally chosen names are `isize/usize`.

# Detailed Design

- Rename `int/uint` to `isize/usize`, with them being their own literal suffixes.
- Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

`usize` in action:

```rust
fn slice_or_fail<'b>(&'b self, from: &usize, to: &usize) -> &'b [T]
```

There are different opinions about which literal suffixes to use. The following section would discuss the alternatives.

## Choosing literal suffixes:

### `isize/usize`:

* Pros: They are the same as the type names, very consistent with the rest of the integer primitives.
* Cons: They are too long for some, and may stand out too much as suffixes. However, discouraging people from overusing `isize/usize` is the point of this RFC. And if they are not overused, then this will not be a problem in practice.

### `is/us`:

* Pros: They are succinct as suffixes.
* Cons: They are actual English words, with `is` being a keyword in many programming languages and `us` being an abbreviation of "unsigned" (losing information) or "microsecond" (misleading). Also, `is/us` may be *too* short (shorter than `i64/u64`) and *too* pleasant to use, which can be a problem.

Note: No matter which suffixes get chosen, it can be beneficial to reserve `is` as a keyword, but this is outside the scope of this RFC.

### `iz/uz`:

* Pros and cons: Similar to those of `is/us`, except that `iz/uz` are not actual words, which is an additional advantage. However it may not be immediately clear that `iz/uz` are abbreviations of `isize/usize`.

### `i/u`:

* Pros: They are very succinct.
* Cons: They are *too* succinct and carry the "default integer types" connotation, which is undesirable. 

### `isz/usz`:

* Pros: They are the middle grounds between `isize/usize` and `is/us`, neither too long nor too short. They are not actual English words and it's clear that they are short for `isize/usize`.
* Cons: Not everyone likes the appearances of `isz/usz`, but this can be said about all the candidates.

After community discussions, it is deemed that using `isize/usize` directly as suffixes is a fine choice and there is no need to introduce other suffixes.

## Advantages of `isize/usize`:

- The names indicate their common use cases (container sizes/indices/offsets), so people will know where to use them, instead of overusing them everywhere.
- The names follow the `i/u + {suffix}` pattern that is used by all the other primitive integer types like `i32/u32`.
- The names are newcomer friendly and have familiarity advantage over almost all other alternatives.
- The names are easy on the eyes.

See **Alternatives B to L** for the alternatives to `isize/usize` that have been rejected.

# Drawbacks

## Drawbacks of the renaming in general:

- Renaming `int`/`uint` requires changing much existing code. On the other hand, this is an ideal opportunity to fix integer portability bugs.

## Drawbacks of `isize/usize`:

- The names fail to indicate the precise semantics of the types - *pointer-sized integers*. (And they don't follow the `i32/u32` pattern as faithfully as possible, as `32` indicates the exact size of the types, but `size` in `isize/usize` is vague in this aspect.)
- The names favour some of the types' use cases over the others.
- The names remind people of C's `ssize_t/size_t`, but `isize/usize` don't share the exact same semantics with the C types.

Familiarity is a double edged sword here. `isize/usize` are chosen not because they are perfect, but because they represent a good compromise between semantic accuracy, familiarity and code readability. Given good documentation, the drawbacks listed here may not matter much in practice, and the combined familiarity and readability advantage outweighs them all.

# Alternatives

## A. Keep the status quo:

Which may hurt in the long run, especially when there is at least one (would-be?) high-profile language (which is Rust-inspired) taking the opposite stance of Rust.

The following alternatives make different trade-offs, and choosing one would be quite a subjective matter. But they are all better than the status quo.

## B. `iptr/uptr`:

- Pros: "Pointer-sized integer", exactly what they are.
- Cons: C/C++ have `intptr_t/uintptr_t`, which are typically *only* used for storing casted pointer values. We don't want people to confuse the Rust types with the C/C++ ones, as the Rust ones have more typical use cases. Also, people may wonder why all data structures have "pointers" in their method signatures. Besides the "funny-looking" aspect, the names may have an incorrect "pointer fiddling and unsafe staff" connotation there, as `ptr` isn't usually seen in safe Rust code.

In the following snippet:

```rust
fn slice_or_fail<'b>(&'b self, from: &uptr, to: &uptr) -> &'b [T]
```

It feels like working with pointers, not integers.

## C. `imem/umem`:

When originally proposed, `mem`/`m` are interpreted as "memory numbers" (See @1fish2's comment in [RFC PR 464](https://github.com/rust-lang/rfcs/pull/464)):

> `imem`/`umem` are "memory numbers." They're good for indexes, counts, offsets, sizes, etc. As memory numbers, it makes sense that they're sized by the address space.

However this interpretation seems vague and not quite convincing, especially when all other integer types in Rust are named precisely in the "`i`/`u` + `{size}`" pattern, with no "indirection" involved. What is "memory-sized" anyway? But actually, they can be interpreted as **_mem_ory-pointer-sized**, and be a *precise* size specifier just like `ptr`.

- Pros: Types with similar names do not exist in mainstream languages, so people will not make incorrect assumptions.
- Cons: `mem` -> *memory-pointer-sized* is definitely not as obvious as `ptr` -> *pointer-sized*. The unfamiliarity may turn newcomers away from Rust.

Also, for some, `imem/umem` just don't feel like integers no matter how they are interpreted, especially under certain circumstances. In the following snippet:

```rust
fn slice_or_fail<'b>(&'b self, from: &umem, to: &umem) -> &'b [T]
```

`umem` still feels like a pointer-like construct here (from "some memory" to "some other memory"), even though it doesn't have `ptr` in its name.

## D. `intp/uintp` and `intm/uintm`:

Variants of Alternatives B and C. Instead of stressing the `ptr` or `mem` part, they stress the `int` or `uint` part.

They are more integer-like than `iptr/uptr` or `imem/umem` if one knows where to split the words.

The problem here is that they don't strictly follow the `i/u + {size}` pattern, are of different lengths, and the more frequently used type `uintp`(`uintm`) has a longer name. Granted, this problem already exists with `int/uint`, but those two are names that everyone is familiar with.

So they may not be as pretty as `iptr/uptr` or `imem/umem`.

```rust
fn slice_or_fail<'b>(&'b self, from: &uintm, to: &uintm) -> &'b [T]
fn slice_or_fail<'b>(&'b self, from: &uintp, to: &uintp) -> &'b [T]
```

## E. `intx/uintx`:

The original proposed names of this RFC, where `x` means "unknown/variable/platform-dependent".

They share the same problems with `intp/uintp` and `intm/uintm`, while *in addition* failing to be specific enough. There are other kinds of platform-dependent integer types after all (like register-sized ones), so which ones are `intx/uintx`?

## F. `idiff/usize`:

There is a problem with `isize`: it most likely will remind people of C/C++ `ssize_t`. But `ssize_t` is in the POSIX standard, not the C/C++ ones, and is *not for index offsets* according to POSIX. The correct type for index offsets in C99 is `ptrdiff_t`, so for a type representing offsets, `idiff` may be a better name.

However, `isize/usize` have the advantage of being symmetrical, and ultimately, even with a name like `idiff`, some semantic mismatch between `idiff` and `ptrdiff_t` would still exist. Also, for fitting a casted pointer value, a type named `isize` is better than one named `idiff`. (Though both would lose to `iptr`.)

## G. `iptr/uptr` *and* `idiff/usize`:

Rename `int/uint` to `iptr/uptr`, with `idiff/usize` being aliases and used in container method signatures.

This is for addressing the "not enough use cases covered" problem. Best of both worlds at the first glance.

`iptr/uptr` will be used for storing casted pointer values, while `idiff/usize` will be used for offsets and sizes/indices, respectively.

`iptr/uptr` and `idiff/usize` may even be treated as different types to prevent people from accidentally mixing their usage.

This will bring the Rust type names quite in line with the standard C99 type names, which may be a plus from the familiarity point of view.

However, this setup brings two sets of types that share the same underlying representations. C distinguishes between `size_t`/`uintptr_t`/`intptr_t`/`ptrdiff_t` not only because they are used under different circumstances, but also because the four may have representations that are potentially different from *each other* on some architectures. Rust assumes a flat memory address space and its `int/uint` types don't exactly share semantics with any of the C types if the C standard is strictly followed.

Thus, even introducing four names would not fix the "failing to express the precise semantics of the types" problem. Rust just doesn't need to, and *shouldn't* distinguish between `iptr/idiff` and `uptr/usize`, doing so would bring much confusion for very questionable gain.

## H. `isiz/usiz`:

A pair of variants of `isize/usize`. This author believes that the missing `e` may be enough to warn people that these are not `ssize_t/size_t` with "Rustfied" names. But at the same time, `isiz/usiz` mostly retain the familiarity of `isize/usize`.

However, `isiz/usiz` still hide the actual semantics of the types, and omitting but a single letter from a word does feel too hack-ish.

```rust
fn slice_or_fail<'b>(&'b self, from: &usiz, to: &usiz) -> &'b [T]
```

## I. `iptr_size/uptr_size`:

The names are very clear about the semantics, but are also irregular, too long and feel out of place.

```rust
fn slice_or_fail<'b>(&'b self, from: &uptr_size, to: &uptr_size) -> &'b [T]
```

## J. `iptrsz/uptrsz`:

Clear semantics, but still a bit too long (though better than `iptr_size/uptr_size`), and the `ptr` parts are still a bit concerning (though to a much less extent than `iptr/uptr`). On the other hand, being "a bit too long" may not be a disadvantage here.

```rust
fn slice_or_fail<'b>(&'b self, from: &uptrsz, to: &uptrsz) -> &'b [T]
```

## K. `ipsz/upsz`:

Now (and only now, which is the problem) it is clear where this pair of alternatives comes from.

By shortening `ptr` to `p`, `ipsz/upsz` no longer stress the "pointer" parts in anyway. Instead, the `sz` or "size" parts are (comparatively) stressed. Interestingly, `ipsz/upsz` look similar to `isiz/usiz`.

So this pair of names actually reflects both the precise semantics of "pointer-sized integers" and the fact that they are commonly used for "sizes". However,

```rust
fn slice_or_fail<'b>(&'b self, from: &upsz, to: &upsz) -> &'b [T]
```

`ipsz/upsz` have gone too far. They are completely incomprehensible without the documentation. Many rightfully do not like letter soup. The only advantage here is that, no one would be very likely to think he/she is dealing with pointers. `iptrsz/uptrsz` are better in the comprehensibility aspect.

## L. Others:

There are other alternatives not covered in this RFC. Please refer to this RFC's comments and [RFC PR 464](https://github.com/rust-lang/rfcs/pull/464) for more.

# Unresolved questions

None. Necessary decisions about Rust's general integer type policies have been made in [Restarting the `int/uint` Discussion](http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131).

# History

Amended by [RFC 573][573] to change the suffixes from `is` and `us` to
`isize` and `usize`. Tracking issue for this amendment is
[rust-lang/rust#22496](https://github.com/rust-lang/rust/issues/22496).

[573]: https://github.com/rust-lang/rfcs/pull/573
