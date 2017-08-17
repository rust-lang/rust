- Feature Name: Unsafe Pointer ~~Reform~~ Methods
- Start Date: 2015-08-01
- RFC PR: [rust-lang/rfcs#1966](https://github.com/rust-lang/rfcs/pull/1966)
- Rust Issue: [rust-lang/rust#43941](https://github.com/rust-lang/rust/issues/43941)


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




## Static Functions

`ptr::foo(ptr)` is an odd interface. Rust developers generally favour the type-directed dispatch provided by methods; `ptr.foo()`. Generally the only reason we've ever shied away from methods is when they would be added to a type that implements Deref generically, as the `.` operator will follow Deref impls to try to find a matching function. This can lead to really confusing compiler errors, or code "spuriously compiling" but doing something unexpected because there was an unexpected match somewhere in the Deref chain. This is why many of Rc's operations are static functions that need to be called as `Rc::foo(&the_rc)`.

This reasoning doesn't apply to the raw pointer types, as they don't provide a Deref impl. Although there are coercions involving the raw pointer types, these coercions aren't performed by the dot operator. This is why it has long been considered fine for raw pointers to have the `deref` and `as_ref` methods.

In fact, the static functions are sometimes useful precisely because they *do* perform raw pointer coercions, so it's possible to do `ptr::read(&val)`, rather than `ptr::read(&val as *const _)`.

However these static functions are fairly cumbersome in the common case, where you already have a raw pointer.




## Signed Offset

The cast in `ptr.offset(idx as isize)` is unnecessarily annoying. Idiomatic Rust code uses unsigned offsets, but low level code is forced to constantly cast those offsets. To understand why this interface is designed as it is, some background is neeeded.

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

* If you offset a `*const i16` by `isize::MAX / 3 * 2` (which fits into a signed integer), then you'll still overflow a signed integer in the implicit `offset` computation.
* There's no indication that unsigned overflow should be a concern at all.
* The location of the offset *isn't even* the place to handle this issue. The ultimate consequence of `offset` being signed is that LLVM can't support allocations larger than `isize::MAX` bytes. Therefore this issue should be handled at the level of memory allocation code.
* The fact that `offset` is `unsafe` is already surprising to anyone with the "it's just addition" mental model, pushing them to read the documentation and learn the actual rules.

In conclusion: `as isize` doesn't help developers write better code.




# Detailed design
[design]: #detailed-design


## Methodization

Add the following method equivalents for the static `ptr` functions on `*const T` and `*mut T`:

(Note that this proposal doesn't deprecate the static functions, as they still make some code more ergonomic than methods, and we'd like to avoid regressing the ergonomics of any usecase. More discussion can be found in the alternatives.)

```rust
impl<T> *(const|mut) T {
  unsafe fn read(self) -> T;
  unsafe fn read_volatile(self) -> T;
  unsafe fn read_unaligned(self) -> T;

  unsafe fn copy_to(self, dest: *mut T, count: usize);
  unsafe fn copy_to_nonoverlapping(self, dest: *mut T, count: usize);
  unsafe fn copy_from(self, src: *mut T, count: usize);
  unsafe fn copy_from_nonoverlapping(self, src: *mut T, count: usize);
}
```

And these only on `*mut T`:

```rust
impl<T> *mut T {
  // note that I've moved these from both to just `*mut T`, to go along with `copy_from`
  unsafe fn drop_in_place(self) where T: ?Sized;
  unsafe fn write(self, val: T);
  unsafe fn write_bytes(self, val: u8, count: usize);
  unsafe fn write_volatile(self, val: T);
  unsafe fn write_unaligned(self, val: T);
  unsafe fn replace(self, val: T) -> T;
  unsafe fn swap(self, with: *mut T);
}
```

(see the alternatives for why we provide both copy_to and copy_from)


## Unsigned Offset

Add the following conveniences to both `*const T` and `*mut T`:

```rust
impl<T> *(const|mut) T {
  unsafe fn add(self, offset: usize) -> Self;
  unsafe fn sub(self, offset: usize) -> Self;
  fn wrapping_add(self, offset: usize) -> Self;
  fn wrapping_sub(self, offset: usize) -> Self;
}
```

I expect `ptr.add` to replace ~95% of all uses of `ptr.offset`, and `ptr.sub` to replace ~95% of the remaining 5%. It's very rare to have an offset that you don't know the sign of, and *also* don't need special handling for.





# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Docs should be updated to use the new methods over the old ones, pretty much
unconditionally. Otherwise I don't think there's anything to do there.

All the docs for these methods can be basically copy-pasted from the existing
functions they're wrapping, with minor tweaks.




# Drawbacks
[drawbacks]: #drawbacks

The only drawback I can think of is that this introduces a "what is idiomatic" schism between the old functions and the new ones.





# Alternatives
[alternatives]: #alternatives


## Overload operators for more ergonomic offsets

Rust doesn't support "unsafe operators", and `offset` is an unsafe function because of the semantics of GetElementPointer. We could make `wrapping_add` be the implementation of `+`, but almost no code should actually be using wrapping offsets, so we shouldn't do anything to make it seem "preferred" over non-wrapping offsets.

Beyond that, `(ptr + idx).read_volatile()` is a bit wonky to write -- methods chain better than operators.




## Make `offset` generic

We could make `offset` generic so it accepts `usize` and `isize`. However we would still want the `sub` method, and at that point we might as well have `add` for symmetry. Also `add` is shorter which is a nice carrot for users to migrate to it.




## Only one of `copy_to` or `copy_from`

`copy` is the only mutating `ptr` operation that doesn't write to the *first* argument. In fact, it's clearly backwards compared to C's memcpy. Instead it's ordered in analogy to `fs::copy`.

Methodization could be an opportunity to "fix" this, and reorder the arguments, providing only `copy_from`. However there is concern that this will lead to users doing a blind migration without checking argument order.

One possibly solution would be deprecating `ptr::copy` along with this as a "signal" that something strange has happened. But as discussed in the following section, immediately deprecating an API along with the introduction of its replacement tends to cause a mess in the broader ecosystem.

On the other hand, `copy_to` isn't as idiomatic (see: `clone_from`), and there was disastisfaction in reinforcing this API design quirk.

As a compromise, we opted to provide both, forcing users of `copy` to decided which they want. Ideally this will be copy_from with reversed arguments, as this is more idiomatic. Longterm we can look to deprecating `copy_to` and `ptr::copy` if desirable. Otherwise having these duplicate methods isn't a big deal (and is *technically* a bit more convenient for users with a reference and a raw pointer).






## Deprecate the Static Functions

To avoid any issues with the methods and static functions coexisting, we could deprecate the static functions. As noted in the motivation, these functions are currently useful for their ability to perform coercions on the first argument. However those who were taking advantage of this property can easily rewrite their code to either of the following:

```
(ptr as *mut _).foo();
<*mut _>::foo(ptr);
```

I personally consider this a minor ergonomic and readability regression from `ptr::foo(ptr)`, and so would rather not do this.

More importantly, this would cause needless churn for old code which is still perfectly *fine*, if a bit less ergonomic than it could be. More ergonomic interfaces should be adopted based on their own merits; not because This Is The New Way, And Everyone Should Do It The New Way.

In fact, even if we decide we should deprecate these functions, we should still stagger the deprecation out several releases to minimize ecosystem churn. When a deprecation occurs, users of the latest compiler will be pressured by diagnostics to update their code to the new APIs. If those APIs were introduced in the same release, then they'll be making their library only compile on the latest release, effectively breaking the library for anyone who hasn't had a chance to upgrade yet. If the deprecation were instead done several releases later, then by the time users are pressured to use the new APIs there will be a buffer of several stable releases that can compile code using the new APIs.


# Unresolved questions
[unresolved]: #unresolved-questions

None.
