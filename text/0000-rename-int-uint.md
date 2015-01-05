- Start Date: 2014-12-28
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

This RFC proposes that we rename the pointer-sized integer types `int/uint`, so as to avoid misconceptions and misuses.

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

However, given the discussions about the previous revisions of this RFC, and the discussions in [Restarting the `int/uint` Discussion]( http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131), this RFC author (@CloudiDust) now believes that `intx/uintx` are not ideal. Instead, one of the other pairs of alternatives should be chosen.

# Detailed Design

Rename `int/uint` to one of the following pairs of alternatives, and decide how to name the literal suffixes for pointer-sized integers based on the selected alternative.  

Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

See **Alternatives B to K** for the alternatives.

# Drawbacks

- Renaming `int`/`uint` requires changing much existing code. On the other hand, this is an ideal opportunity to fix integer portability bugs.

# Alternatives

## A. Keep the status quo.

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

## F. `idiff(isize)/usize`:

Previously, the names involving suffixes like `diff`/`addr`/`size`/`offset` are rejected mainly because they favour specific use cases of `int/uint` while overlooking others. However, it is true that in the majority of cases in safe code, Rust's `int/uint` are used just like standard C/C++ `ptrdiff_t/size_t`. When used in this context, names `idiff(isize)/usize` have clarity and familiarity advantages **over all other alternatives**.

(Note: this author advices against `isize`, as it most likely corresponds to C/C++ `ssize_t`. `ssize_t` is in the POSIX standard, not the C/C++ ones, and is *not for offsets* according to that standard. However some may argue that, `isize/usize` are different enough from `ssize_t/size_t` so this author's worries are unnecessary.)

```rust
fn slice_or_fail<'b>(&'b self, from: &usize, to: &usize) -> &'b [T]
```

But how about the other use cases of `int/uint` especially the "storing casted pointers" one? Using `libc`'s `intptr_t`/`uintptr_t` is not an option here, as "Rust on bare metal" would be ruled out. Forcing a pointer value into something called `idiff/usize` doesn't seem right either. Thus, this leads us to:

## G. `iptr/uptr` *and* `idiff/usize`:

Rename `int/uint` to `iptr/uptr`, with `idiff/usize` being aliases and used in container method signatures.

Best of both worlds on the first glance.

`iptr/uptr` will be used for storing casted pointer values, while `idiff/usize` will be used for offsets and sizes/indices, respectively.

`iptr/uptr` and `idiff/usize` may even be treated as different types to prevent people from accidentally mixing their usage.

This will bring the Rust type names quite in line with the standard C99 type names, which may be a plus from the familiarity point of view.

However, this setup brings two sets of types that share the same underlying representations, which also brings confusion. Furthermore, C distinguishes between `size_t`/`uintptr_t`/`intptr_t`/`ptrdiff_t` not only because they are used under different circumstances, but also because the four may have representations that are potentially different from *each other* on some architectures. Rust assumes a flat memory address space and its `int/uint` types don't exactly share semantics with any of the C types if the C standard is strictly followed. Thus, this RFC author believes that, it is better to completely forego type names that will remind people of the C types.

## H. `isiz/usiz`:

A pair of variants of `isize/usize`. This author believes that the missing `e` may be enough to warn people that these are not `ssize_t/size_t` with "Rustfied" names. But at the same time, `isiz/usiz` mostly retain the familiarity of `isize/usize`. Actually, this author considers them more pleasant to use than the "full version".

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

Now (and only now, which is the problem) it is clear where this final pair of alternatives comes from.

By shortening `ptr` to `p`, `ipsz/upsz` no longer stress the "pointer" parts in anyway. Instead, the `sz` or "size" parts are (comparatively) stressed. Interestingly, `ipsz/upsz` look similar to `isiz/usiz`.

So this pair of names actually reflects both the precise semantics of "pointer-sized integers" and the fact that they are commonly used for "sizes". However,

```rust
fn slice_or_fail<'b>(&'b self, from: &upsz, to: &upsz) -> &'b [T]
```

`ipsz/upsz` have gone too far. They are completely incomprehensible without the documentation. Many rightfully do not like letter soup. The only advantage here is that, no one would be very likely to think he/she is dealing with pointers. `iptrsz/uptrsz` are better in this regard.

# Unresolved questions

None. Necessary decisions about Rust's general integer type policies have been made in [Restarting the `int/uint` Discussion](http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131).
