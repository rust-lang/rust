- Start Date: 2014-12-28
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

This RFC proposes that we rename the pointer-sized integer types `int/uint` to `intp/uintp`, `intm/uintm` or `imem/umem`, so as to avoid misconceptions and misuses.

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

As stated in previous discussions, all suggested alternative names have some drawbacks that may be unbearable. (Please refer to [A tale of two's complement](http://discuss.rust-lang.org/t/a-tale-of-twos-complement/1062) and related discussions for details.)

Before the rejection, the community largely settled on two pairs of candidates: `imem/umem` and `iptr/uptr`.

`iptr/uptr` were rejected because they may remind the programmers of C/C++ `intptr_t`/`uintptr_t`, which were typically *only* used for storing casted pointer values. However, favouring any one of the types' use cases in the names is undesirable.

`imem/umem` were rejected because they may be "not integer-like" and reminded people of "int memory"/"unsigned int memory" which made no sense.

But the problems can be dealt with, in one of the following ways:

1. instead of stressing the `ptr`/`mem` part, stress the `i`/`u` part,
2. or find and canonicalize a better interpretation for `imem/umem`.

# Detailed Design

## Approach 1.

Rename `int/uint` to `intp/uintp` or `intm/uintm`.

Introduce the respective literal suffixes `ip/up` or `im/um`.

Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

## Approach 2.

Rename `int/uint` to `imem/umem`, and use `imem/umem` as the new literal suffixes.

Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

How to deal with the problems of `imem/umem`?

Interpret them in the documentation as: **signed/unsigned _mem_ory-pointer-sized integers.**

With this interpretation in mind, the RFC author (@CloudiDust) considers `imem`/`umem` to once again be his favourite among the three pairs of candidates. 

## Advantages

The advantages of both approaches over `int`/`uint`:

- The names are foreign to programmers from other languages, so they are less likely to make incorrect assumptions, or use them out of habit.
- But not too foreign, they still look like integers. (Depending on personal taste, Approach 1 is either better or worse than Approach 2 on this one.)
- They follow the same *signed-ness + size* naming pattern used by other integer types like `i32/u32`. (Approach 2 is arguably better here.)

# Drawbacks

Renaming `int`/`uint` requires changing much existing code. On the other hand, this is an ideal opportunity to fix integer portability bugs.

`intp/uintp` are supported by several community members, but they still may remind people of `intptr_t`/`uintptr_t`, although arguably to a lesser extent, as `p` can be interpreted as `platform-dependent` here.

With different trade-offs considered, `intm/uintm` may or may not be a better name than `imem/umem` depending on personal tastes.

# Alternatives

## A. Keep the status quo.

Which may hurt in the long run, especially when there is at least one (would-be?) high-profile language (which is Rust-inspired) taking the opposite stance of Rust.

## B. Approach 1a: Use `intx/uintx` as the new type names, and `ix/ux` as the new suffixes.

`intx/uintx` were actually the names that got promoted in the previous revisions of this RFC, where `x` means "unknown", "variable" or "platform-dependent". However the `x` suffix was too vague as there were other integer types that have platform-dependent sizes, like register-sized ones, so `intx/uintx` lost their "promoted" status in this revision.

## C. Approach 1b: Use the proposed literal suffixes as the new type names, not just literal suffixes.

While that will make type names more closely follow the `i32/u32` pattern, they may be too short (and tempting) and may have unrelated meanings on their own (`ip`/`up`/`um` etc.)  

## D. Approach 1c: Use the proposed type names as literal suffixes, not just type names.

`uintp`/`uintm`/`uintx` may or may not too long as literal suffixes.

# Unresolved questions

None.
