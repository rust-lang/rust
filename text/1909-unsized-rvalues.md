- Feature Name: unsized_locals
- Start Date: 2017-02-11
- RFC PR: [rust-lang/rfcs#1909](https://github.com/rust-lang/rfcs/pull/1909)
- Rust Issue: [rust-lang/rust#48055](https://github.com/rust-lang/rust/issues/48055)

# Summary
[summary]: #summary

Allow for local variables, function arguments, and some expressions to have an unsized type, and implement it by storing the temporaries in variably-sized allocas.

Have repeat expressions with a length that captures local variables be such an expression, returning an `[T]` slice.

Provide some optimization guarantees that unnecessary temporaries will not create unnecessary allocas.

# Motivation
[motivation]: #motivation

There are 2 motivations for this RFC:

1. Passing unsized values, such as trait objects, to functions by value is often desired. Currently, this must be done through a `Box<T>` with an unnecessary allocation.

  One particularly common example is passing closures that consume their environment without using monomorphization. One would like for this code to work:

  ```Rust
  fn takes_closure(f: FnOnce()) { f(); }
  ```

  But today you have to use a hack, such as taking a `Box<FnBox<()>>`.

2. Allocating a runtime-sized variable on the stack is important for good performance in some use-cases - see RFC #1808, which this is intended to supersede.

# Detailed design
[design]: #detailed-design

## Unsized Rvalues - language

Remove the rule that requires all locals and rvalues to have a sized type. Instead, require the following:

1. The following expressions must always return a Sized type:
    1. Function calls, method calls, operator expressions
        - implementing unsized return values for function calls would require the *called function* to do the alloca in our stack frame.
    2. ADT expressions
        - see alternatives
    3. cast expressions
        - this seems like an implementation simplicity thing. These can only be trivial casts.
2. The RHS of assignment expressions must always have a Sized type.
    - Assigning an unsized type is impossible because we don't know how much memory is available at the destination. This applies to ExprAssign assignments and not to StmtLet let-statements.

This also allows passing unsized values to functions, with the ABI being as if a `&move` pointer was passed (a `(by-move-data, extra)` pair). This also means that methods taking `self` by value are object-safe, though vtable shims are sometimes needed to translate the ABI (as the callee-side intentionally does not pass `extra` to the fn in the vtable, no vtable shim is needed if the vtable function already takes its argument indirectly).

For example:

```Rust
struct StringData {
    len: usize,
    data: [u8],
}

fn foo(s1: Box<StringData>, s2: Box<StringData>, cond: bool) {
    // this creates a VLA copy of either `s1.1` or `s2.1` on
    // the stack.
    let mut s = if cond {
        s1.data
    } else {
        s2.data
    };
    drop(s1);
    drop(s2);
    foo(s);
}

fn example(f: for<'a> FnOnce(&'a X<'a>)) {
    let x = X::new();
    f(x); // aka FnOnce::call_once(f, (x,));
}
```

## VLA expressions

Allow repeat expressions to capture variables from their surrounding environment. If a repeat expression captures such a variable, it has type `[T]` with the length being evaluated at run-time. If the repeat expression does not capture any variable, the length is evaluated at compile-time. For example:
```Rust
extern "C" {
   fn random() -> usize;
}

fn foo(n: usize) {
    let x = [0u8; n]; // x: [u8]
    let x = [0u8; n + (random() % 100)]; // x: [u8]
    let x = [0u8; 42]; // x: [u8; 42], like today
    let x = [0u8; random() % 100]; //~ ERROR constant evaluation error
}
```
"captures a variable" - as in RFC #1558 - is used as the condition for making the return be `[T]` because it is simple, easy to understand, and  introduces no type-checking complications.

The last error message could have a user-helpful note, for example "extract the length to a local variable if you want a variable-length array".

## Unsized Rvalues - MIR

The way this is implemented in MIR is that operands, rvalues, and temporaries are allowed to be unsized. An unsized operand is always "by-ref". Unsized rvalues are either a `Use` or a `Repeat` and both can be translated easily.

Unsized locals can never be reassigned within a scope. When first assigning to an unsized local, a stack allocation is made with the correct size.

MIR construction remains unchanged. 

## Guaranteed Temporary Elision

MIR likes to create lots of temporaries for OOE reason. We should optimize them out in a guaranteed way in these cases (FIXME: extend these guarantees to locals aka NRVO?).

TODO: add description of problem & solution.
    
# How We Teach This
[teach]: #how-we-teach-this

Passing arguments to functions by value should not be too complicated to teach. I would like VLAs to be mentioned in the book.

The "guaranteed temporary elimination" rules require more work to teach. It might be better to come up with new rules entirely.

# Drawbacks
[drawbacks]: #drawbacks

In Unsafe code, it is very easy to create unintended temporaries, such as in:
```Rust
unsafe fn poke(ptr: *mut [u8]) { /* .. */ }
unsafe fn foo(mut a: [u8]) {
    let ptr: *mut [u8] = &mut a;
    // here, `a` must be copied to a temporary, because
    // `poke(ptr)` might access the original.
    bar(a, poke(ptr));
}
```

If we make `[u8]` be `Copy`, that would be even easier, because even uses of `poke(ptr);` after the function call could potentially access the supposedly-valid data behind `a`.

And even if it is not as easy, it is possible to accidentally create temporaries in safe code.

Unsized temporaries are dangerous - they can easily cause aborts through stack overflow.

# Alternatives
[alternatives]: #alternatives

## The bikeshed

There are several alternative options for the VLA syntax.

1. The RFC choice, `[t; φ]` has type `[T; φ]` if `φ` captures no variables and type `[T]` if φ captures a variable.
    - pro: can be understood using "HIR"/resolution only.
    - pro: requires no additional syntax.
    - con: might be confusing at first glance.
    - con: `[t; foo()]` requires the length to be extracted to a local.
2. The "permissive" choice: `[t; φ]` has type `[T; φ]` if `φ` is a constexpr, otherwise `[T]`
    - pro: allows the most code
    - pro: requires no additional syntax.
    - con: depends on what is exactly a const expression. This is a big issue because that is both non-local and might change between rustc versions.
3. Use the expected type - `[t; φ]` has type `[T]` if it is evaluated in a context that expects that type (for example `[t; foo()]: [T]`) and `[T; _]` otherwise.
    - pro: in most cases, very human-visible.
    - pro: requires no additional syntax.
    - con: relies on the notion of "expected type". While I think we *do* have to rely on that in the unsafe code semantics of `&foo` borrow expressions (as in, whether a borrow is treated as a "safe" or "unsafe" borrow - I'll write more details sometime), it might be better to not rely on expected types too much.
4. use an explicit syntax, for example `[t; virtual φ]`.
    - bikeshed: exact syntax.
    - pro: very explicit and visible.
    - con: more syntax.
5. use an intrinsic, `std::intrinsics::repeat(t, n)` or something.
    - pro: theoretically minimizes changes to the language.
    - con: requires returning unsized values from intrinsics.
    - con: unergonomic to use.

## Unsized ADT Expressions

Allowing unsized ADT expressions would make unsized structs constructible without using unsafe code, as in:
```Rust
let len_ = s.len();
let p = Box::new(PascalString {
    length: len_,
    data: *s
});
```

However, without some way to guarantee that this can be done without allocas, that might be a large footgun.

## Copy Slices

One somewhat-orthogonal proposal that came up was to make `Clone` (and therefore `Copy`) not depend on `Sized`, and to make `[u8]` be `Copy`, by moving the `Self: Sized` bound from the trait to the methods, i.e. using the following declaration:
```Rust
pub trait Clone {
    fn clone(&self) -> Self where Self: Sized;
    fn clone_from(&mut self, source: &Self) where Self: Sized {
        // ...
    }
}
```

That would be a backwards-compatability-breaking change, because today `T: Clone + ?Sized` (or of course `Self: Clone` in a trait context, with no implied `Self: Sized`) implies that `T: Sized`, but it might be that its impact is small enough to allow (and even if not, it might be worth it for Rust 2.0).

# Unresolved questions
[unresolved]: #unresolved-questions

How can we mitigate the risk of unintended unsized or large allocas? Note that the problem already exists today with large structs/arrays. A MIR lint against large/variable stack sizes would probably help users avoid these stack overflows. Do we want it in Clippy? rustc?

How do we handle truely-unsized DSTs when we get them? They can theoretically be passed to functions, but they can never be put in temporaries.

Accumulative allocas (aka `'fn` borrows) are beyond the scope of this RFC.

See alternatives.
