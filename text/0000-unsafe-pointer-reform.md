- Feature Name: Unsafe Pointer ~~Reform~~ Methods
- Start Date: 2015-08-01
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)


# Summary
[summary]: #summary

Copy most of the static `ptr::` functions to methods on unsafe pointers themselves.
Also add a few conveniences for `ptr.offset` with unsigned integers.

```rust
// So this:
ptr::read(self.ptr.offset(idx as isize))

// Becomes this:
self.ptr.add(idx).read()
```

More conveniences should probably be added to unsafe pointers, but this proposal is basically the "minimally controversial" conveniences.




# Motivation
[motivation]: #motivation


Swift lets you do this:

```swift
let val = ptr.advanced(by: idx).move()
```

And we want to be cool like Swift, right?




## Static Functions Stink

`ptr::foo(ptr)` is unnecessarily annoying, and requires imports. That's it.

The static functions are slightly useful because they can be more convenient in cases where you have a safe pointer. Specifically, they act as a coercion site so `ptr::read(&my_val)` works, and is nicer than `(&my_val as *const _).read()`. But if you already have unsafe pointers, which is the common case, it's a worse interface.




## Signed Offset Stinks

The cast in `ptr.offset(idx as isize)` is unnecessarily annoying. Idiomatic Rust code uses unsigned offsets, but the low level code has to constantly cast those offsets  This one requires more detail to explain. 

`offset` is directly exposing LLVM's `getelementptr` instruction, with the `inbounds` keyword. `wrapping_offset` removes the `inbounds` keyword. `offset` takes a signed integer, because that's what GEP exposes. It's understandable that we've been conservative here; GEP is so confusing that it has an [entire FAQ](http://llvm.org/docs/GetElementPtr.html).

That said, LLVM is pretty candid that it models pointers as two's complement integers, and a negative integer is just a really big positive integer, right? So can we provide an unsigned version of offset, and just feed it down into GEP?

[The relevant FAQ entry](http://llvm.org/docs/GetElementPtr.html#what-happens-if-a-gep-computation-overflows) is as follows:

> What happens if a GEP computation overflows?
>
> If the GEP lacks the inbounds keyword, the value is the result from evaluating the implied two’s complement integer computation. However, since there’s no guarantee of where an object will be allocated in the address space, such values have limited meaning.
>
> If the GEP has the inbounds keyword, the result value is undefined (a “trap value”) if the GEP overflows (i.e. wraps around the end of the address space).
>
> As such, there are some ramifications of this for inbounds GEPs: scales implied by array/vector/pointer indices are always known to be “nsw” since they are signed values that are scaled by the element size. These values are also allowed to be negative (e.g. “`gep i32 *%P, i32 -1`”) but the pointer itself is logically treated as an unsigned value. This means that GEPs have an asymmetric relation between the pointer base (which is treated as unsigned) and the offset applied to it (which is treated as signed). The result of the additions within the offset calculation cannot have signed overflow, but when applied to the base pointer, there can be signed overflow.

This is written in a bit of a confusing way, so here's a simplified summary of what we care about: 

* The pointer is treated as an unsigned number, and the offset as signed. 
* While computing the offset in bytes (`idx * size_of::<T>()`), we aren't allowed to do signed overflow (nsw). 
* While applying the offset to the pointer (`ptr + offset`), we aren't allowed to do unsigned overflow (nuw).

Part of the historical argument for signed offset in Rust has been a *warning* against these overflow concerns, but upon inspection that doesn't really make sense. 

* If you offset a `*const i16` by `isize::MAX * 3 / 2` (which fits into a signed integer), then you'll still overflow a signed integer in the implicit `offset` computation. 
* There's no indication that unsigned overflow should be a concern at all.
* The location of the offset *isn't even* the place to handle this issue. The ultimate consequence of `offset` being signed is that LLVM can't support allocations larger than `isize::MAX` bytes. Therefore this issue should be handled at the level of memory allocation code.
* The fact that `offset` is `unsafe` is already surprising to anyone with the "it's just addition" mental model, pushing them to read the documentation and learn the actual rules.

In conclusion: `as isize` sucks and isn't helpful; let's just get some unsigned versions of offset!




# Detailed design
[design]: #detailed-design


## Methodization

Add the following method equivalents for the static `ptr` functions on `*const T` and `*mut T`:

```rust
ptr.copy(dst: *mut T, count: usize)
ptr.copy_nonoverlapping(dst: *mut T, count: usize)
ptr.read() -> T
ptr.read_volatile() -> T
ptr.read_unaligned() -> T
```

And these only on `*mut T`:

```rust
ptr.drop_in_place()
ptr.write(val: T)
ptr.write_bytes(val: u8, count: usize)
ptr.write_volatile(val: T)
ptr.write_unaligned()
ptr.replace(val: T) -> T
```

`ptr.swap` has been excluded from this proposal because it's a symmetric operation, and is consequently a bit weird to methodize.

The static functions should remain undeprecated, as they are more ergonomic in the cases explained in the motivation.




## Unsigned Offset

Add the following conveniences to both `*const T` and `*mut T`: 

```rust
ptr.add(offset: usize) -> Self
ptr.sub(offset: usize) -> Self
ptr.wrapping_add(offset: usize) -> Self
ptr.wrapping_sub(offset: usize) -> Self
```

I expect `ptr.add` to replace ~95% of all uses of `ptr.offset`, and `ptr.sub` to replace ~95% of the remaining 5%. It's just very weird to have an offset that you don't know the sign of.





# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Docs should be updated to use the new methods over the old ones, pretty much
unconditionally. Otherwise I don't think there's anything to do there.

All the docs for these methods can be basically copy-pasted from the existing
functions they're wrapping, with minor tweaks.




# Drawbacks
[drawbacks]: #drawbacks

This proposal bloats the stdlib and introduces a schism between the old and new style. That said, the new style is way better, and the bloat is minor, so that's nothing worth worrying about.





# Alternatives
[alternatives]: #alternatives


## Overload operators for more ergonomic offsets

Rust doesn't support "unsafe operators", and `offset` is an unsafe function because of the semantics of GetElementPointer. We don't want `wrapping_add` to get the operator, because `add` is the one we want developers to use. Beyond that, `(ptr + idx).read_volatile()` is a bit wonky to write.



## Make `offset` generic 

You could make `offset` generic so it accepts `usize` and `isize`. However you would still want the `sub` method, and at that point you might as well have `add` for symmetry. Also `add` is shorter which, as we all know, is better.




# Unresolved questions
[unresolved]: #unresolved-questions

Should `ptr::swap` be made into a method? I am personally ambivalent.
