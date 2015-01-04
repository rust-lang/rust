- Start Date: 2014-12-28
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

This RFC proposes that we rename the pointer-sized integer types `int/uint` to stress the fact that they are pointer-sized, so as to avoid misconceptions and misuses. Among all the candidates, this RFC author (@CloudiDust) considers `imem/umem` to be his favourite names.

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

However, given the discussions about the previous revisions of this RFC, and the discussions in [Restarting the `int/uint` Discussion]( http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131), this RFC author now believes that `iptr/uptr` and `imem/umem` can still be viable candidates.

This RFC also proposes two more pairs of candidates: `intp/uintp` and `intm/uintm`, which are actually variants of the above, but stress the `int` part instead of the `ptr`/`mem` part, which may make them subjectively better (or worse), as some would think `iptr/uptr` and `imem/umem` stress the wrong part of the name, and don't look like integer types.

# Detailed Design

Rename `int/uint` to one of the above pairs of candidates.

Update code and documentation to use pointer-sized integers more narrowly for their intended purposes. Provide a deprecation period to carry out these updates.

Also, a decision should be made about whether to introduce the respective literal suffixes `ip/up` or `im/um`, or use the type names as literal suffixes.

This author believes that, if `iptr/uptr` or `imem/umem` are chosen, they should be used directly as suffixes, but if `intp/uintp` or `intm/uintm` are chosen, then `ip/up` or `im/um` should be used, as `uintp`/`uintm` may be too long as suffixes. `ip/up` and `im/um` don't make good *type names* though, as they are too short and have meanings unrelated to integers.

## Advantages

Each pair of candidates make different trade-offs, and choosing one would be a quite subjective matter. But they are all better than `int/uint`.

### The advantages of all candidates over `int/uint`:

- The names are foreign to programmers from other languages, so they are less likely to make incorrect assumptions, or use them out of habit.
- But not too foreign, they still look like integers. (`intp/uintp` and `intm/uintm` may be a bit better here.)
- They follow the same *signed-ness + size* naming pattern used by other integer types like `i32/u32`. (`iptr/uptr` and `imem/umem` are better here as they follow the pattern more faithfully. Please see the following discussion for why `imem/umem` also have the *size* part.)

### The advantages and disadvantages of suffixes `ptr`/`p` and `mem`/`m`:

#### `iptr/uptr` and `intp/uintp`:

- Pros: "Pointer-sized integer", exactly what they are.
- Cons: C/C++ have `intptr_t/uintptr_t`, which are typically *only* used for storing casted pointer values. We don't want people to confuse the Rust types with the C/C++ ones, as the Rust ones have more typical use cases. Also, people may wonder why all data structures have "pointers" in their method signatures. Besides the "funny-looking" aspect, the names may have an incorrect "pointer fiddling and unsafe staff" connotation there.

However, note that Rust (barring FFI) don't have `size_t`/`ptrdiff_t` etc like C/C++ do, so the disadvantage may not actually matter much.

Also, there are talks about parametrizing data structures over their indexing/size types, if this lands, then in a sense the pointer-sized integers would no longer be the "privileged types for sizes/indexes", and `iptr/uptr` or `intp/uintp` would be quite fine names.  

#### `imem/umem` and `intm/uintm`:

When originally proposed, `mem`/`m` are interpreted as "memory numbers" (See @1fish2's comment in[RFC PR 464](https://github.com/rust-lang/rfcs/pull/464)):

> `imem`/`umem` are "memory numbers." They're good for indexes, counts, offsets, sizes, etc. As memory numbers, it makes sense that they're sized by the address space.

However this interpretation seems vague and not quite convincing, especially when all other integer types in Rust are named precisely in the "`i`/`u` + `size`" pattern, with no "indirection" involved. What is "memory-sized" anyway? But actually, they can be interpreted as **_mem_ory-pointer-sized**, and be a *precise* size specifier just like `ptr`.

- Pros: Types with similar names do not exist in mainstream languages, so people will not make incorrect assumptions.
- Cons: `mem` -> *memory-pointer-sized* is definitely not as obvious as `ptr` -> *pointer-sized*.

However, people will be tempted to read the documentation anyway when they encounter `imem/umem` or `intm/uintm`. And this RFC author expects the "memory-pointer-sized" interpretation to be easy (or easier) to internalize once the documentation gets consulted.

### Note:

This RFC author personally prefers `imem/umem` now, but `intp/uintp` and `iptr/uptr` also have plenty of community support. No one else seem to care about `intm/uintm`, and they are only in for symmetry and completeness.

# Drawbacks

Renaming `int`/`uint` requires changing much existing code. On the other hand, this is an ideal opportunity to fix integer portability bugs.

# Alternatives

## A. Keep the status quo.

Which may hurt in the long run, especially when there is at least one (would-be?) high-profile language (which is Rust-inspired) taking the opposite stance of Rust.

## B. Use `intx/uintx` as the new type names.

`intx/uintx` were actually the names that got promoted in the previous revisions of this RFC, where `x` means "unknown", "variable" or "platform-dependent". However the `x` suffix was too vague as there were other integer types that have platform-dependent sizes, like register-sized ones, so `intx/uintx` lost their "promoted" status in this revision.

## C. Use `idiff/usize` as the new type names.

Previously, the names involving suffixes like `diff`/`addr`/`size`/`offset` are rejected mainly because they favour specific use cases of `int/uint` while overlooking others. However, it is true that in the majority of cases in safe code, Rust's `int/uint` are used just like standard C/C++ `ptrdiff_t/size_t`. When used in this context, names `idiff/usize` have clarity and familiarity advantages **over all other alternatives**.

(Note: this author advices against `isize`, as it most likely corresponds to C/C++ `ssize_t`. `ssize_t` is in the POSIX standard, not the C/C++ ones, and is *not for offsets/diffs* according to that standard.)

But how about the other use cases of `int/uint` especially the "storing casted pointers" one? Using `libc`'s `intptr_t`/`uintptr_t` is not an option here, as "Rust on bare metal" would be ruled out. Forcing a pointer value into something called `idiff/usize` doesn't seem right either. Thus, this leads us to:

## D. Rename `int/uint` to `iptr/uptr`, with `idiff/usize` being aliases and preferred in container method signatures.

Best of both worlds, maybe?

`iptr/uptr` will be used for storing casted pointer values, while `idiff/usize` will be used for offsets and sizes/indices, respectively.

We may even treat `iptr/uptr` and `idiff/usize` as different types to prevent people from accidentally mixing their usage.

This will bring the Rust type names quite in line with the standard C99 type names, which may be a plus from the familiarity point of view.

However, this setup brings two sets of types that share the same underlying representations, which also brings confusion. Furthermore, C distinguishes between `size_t`/`uintptr_t`/`intptr_t`/`ptrdiff_t` not only because they are used under different circumstances, but also because the four may have representations that are potentially different from *each other* on some architectures. Rust assumes a flat memory address space and its `int/uint` types don't exactly share semantics with any of the C types if the C standard is strictly followed. Thus, this RFC author believes that, it is better to completely forego type names that will remind people of the C types.

`imem/umem` are still the better choices.

# Unresolved questions

This RFC author believes that we should nail down our general integer type/coercion/data structure indexing policies, as discussed in [Restarting the `int/uint` Discussion](http://discuss.rust-lang.org/t/restarting-the-int-uint-discussion/1131), before deciding which names to rename `int/uint` to.
