- Start Date: 2014-12-28
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Rename the pointer-sized integer types `int/uint` to `intx/uintx`, and use new literal suffixes `ix/ux`, so as to avoid misconceptions and misuses.

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

Therefore this RFC proposes a new pair of alternatives: `intx/uintx`, where the `x` suffix means "unknown size"/"variable size", or "platform-dependent size".

The pros:

- The names are foreign to programmers from other languages, so they are less likely to make incorrect assumptions, or use them out of habit.
- But not too foreign, they still look like integer type names. (Some believe that `imem/umem` fail here.)
- They do not favour one of the types' use cases over the others in the names. (Alternatives `iptr/uptr`, `idiff/usize` and others fail here.)
- They follow the same *signed-ness + size* naming pattern used by other integer types like `i32/u32`.
- They somewhat look like `index/uindex`. This may or may not be an advantage.

# Detailed Design

Rename these two pointer-sized integer types, `int` to `intx`, and `uint` to `uintx`.

Use `ix` and `ux` as the literal suffix for `intx` and `uintx`, respectively.

Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

# Drawbacks

- Renaming `int`/`uint` requires changing much existing code. On the other hand, this is an ideal opportunity to fix integer portability bugs.
- The new names are longer (but not much longer).
- The `x` suffix may be too generic and doesn't carry enough meaning. In particular, it signifies the fact that the size is "unknown"/"variable" "in some way", but what is this "some way" after all?

# Alternatives

**A. Keep the status quo.**

Which may hurt in the long run, especially when there is at least one (would-be?) high-profile language (which is Rust-inspired) taking the opposite stance of Rust.

**B. Use `ix/ux` as the new type names, not just literal suffixes.**

While `ix/ux` more closely follow the `i32/u32` pattern, they may be too short (and tempting) and may not look like integer types for some.

**C. Use `intx/uintx` as the new literal suffixes, not just type names.**

For some, `42intx/42uintx` are too long and don't look pretty, but then again others may find this desirable.

**D. Use `intp/uintp` and/or `ip/up` instead.**

Here `p` means "pointer (sized)" or "platform (dependent)", thus making the semantics of `intp/uintp` clearer than that of `intx/uintx`.

The drawback here is that some people may incorrectly assume that `intp/uintp` *only* have the same use case as C/C++'s `intptr_t/uintptr_t`, which are *only* for storing casted pointer values.

Also, as literal suffixes or type names, `ip/up` may be more confusing than `ix/ux`, as `ip/up` have meanings that aren't related to integers.

**E. Use `imem/umem` and/or `im/um` instead.**

While `imem/umem` was rejected previously, it is still controversial whether they are truly "ugly" or "not integer-like". Also, they may have some advantages over `intx/uintx`:

- They actually more closely follow the `i32/u32` pattern: `i/u` + **mem**ory pointer-sized.
- So they also better describe what size they have, instead of just stating "unknown"/"variable", but the unfortunate implications of `intp/uintp` are avoided.
- If one prefers `imem/umem` as type names, then they also make better suffixes than `intx`/`uintx` because `umem` is shorter than `uintx` and `imem/umem` are of the same length.

`im/um` may also be more (or less) confusing than `ix/ux`.

A related pair of variants `intm/uintm` may also be worth considering.

# Unresolved questions

None.
