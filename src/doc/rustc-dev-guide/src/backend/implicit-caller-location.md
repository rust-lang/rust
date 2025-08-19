# Implicit caller location

Approved in [RFC 2091], this feature enables the accurate reporting of caller location during panics
initiated from functions like `Option::unwrap`, `Result::expect`, and `Index::index`. This feature
adds the [`#[track_caller]`][attr-reference] attribute for functions, the
[`caller_location`][intrinsic] intrinsic, and the stabilization-friendly
[`core::panic::Location::caller`][wrapper] wrapper.

## Motivating example

Take this example program:

```rust
fn main() {
    let foo: Option<()> = None;
    foo.unwrap(); // this should produce a useful panic message!
}
```

Prior to Rust 1.42, panics like this `unwrap()` printed a location in core:

```
$ rustc +1.41.0 example.rs; example.exe
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value',...core\macros\mod.rs:15:40
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace.
```

As of 1.42, we get a much more helpful message:

```
$ rustc +1.42.0 example.rs; example.exe
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', example.rs:3:5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

These error messages are achieved through a combination of changes to `panic!` internals to make use
of `core::panic::Location::caller` and a number of `#[track_caller]` annotations in the standard
library which propagate caller information.

## Reading caller location

Previously, `panic!` made use of the `file!()`, `line!()`, and `column!()` macros to construct a
[`Location`] pointing to where the panic occurred. These macros couldn't be given an overridden
location, so functions which intentionally invoked `panic!` couldn't provide their own location,
hiding the actual source of error.

Internally, `panic!()` now calls [`core::panic::Location::caller()`][wrapper] to find out where it
was expanded. This function is itself annotated with `#[track_caller]` and wraps the
[`caller_location`][intrinsic] compiler intrinsic implemented by rustc. This intrinsic is easiest
explained in terms of how it works in a `const` context.

## Caller location in `const`

There are two main phases to returning the caller location in a const context: walking up the stack
to find the right location and allocating a const value to return.

### Finding the right `Location`

In a const context we "walk up the stack" from where the intrinsic is invoked, stopping when we
reach the first function call in the stack which does *not* have the attribute. This walk is in
[`InterpCx::find_closest_untracked_caller_location()`][const-find-closest].

Starting at the bottom, we iterate up over stack [`Frame`][const-frame]s in the
[`InterpCx::stack`][const-stack], calling
[`InstanceKind::requires_caller_location`][requires-location] on the
[`Instance`s from each `Frame`][frame-instance]. We stop once we find one that returns `false` and
return the span of the *previous* frame which was the "topmost" tracked function.

### Allocating a static `Location`

Once we have a `Span`, we need to allocate static memory for the `Location`, which is performed by
the [`TyCtxt::const_caller_location()`][const-location-query] query. Internally this calls
[`InterpCx::alloc_caller_location()`][alloc-location] and results in a unique
[memory kind][location-memory-kind] (`MemoryKind::CallerLocation`). The SSA codegen backend is able
to emit code for these same values, and we use this code there as well.

Once our `Location` has been allocated in static memory, our intrinsic returns a reference to it.

## Generating code for `#[track_caller]` callees

To generate efficient code for a tracked function and its callers, we need to provide the same
behavior from the intrinsic's point of view without having a stack to walk up at runtime. We invert
the approach: as we grow the stack down we pass an additional argument to calls of tracked functions
rather than walking up the stack when the intrinsic is called. That additional argument can be
returned wherever the caller location is queried.

The argument we append is of type `&'static core::panic::Location<'static>`. A reference was chosen
to avoid unnecessary copying because a pointer is a third the size of
`std::mem::size_of::<core::panic::Location>() == 24` at time of writing.

When generating a call to a function which is tracked, we pass the location argument the value of
[`FunctionCx::get_caller_location`][fcx-get].

If the calling function is tracked, `get_caller_location` returns the local in
[`FunctionCx::caller_location`][fcx-location] which was populated by the current caller's caller.
In these cases the intrinsic "returns" a reference which was actually provided in an argument to its
caller.

If the calling function is not tracked, `get_caller_location` allocates a `Location` static from
the current `Span` and returns a reference to that.

We more efficiently achieve the same behavior as a loop starting from the bottom by passing a single
`&Location` value through the `caller_location` fields of multiple `FunctionCx`s as we grow the
stack downward.

### Codegen examples

What does this transformation look like in practice? Take this example which uses the new feature:

```rust
#![feature(track_caller)]
use std::panic::Location;

#[track_caller]
fn print_caller() {
    println!("called from {}", Location::caller());
}

fn main() {
    print_caller();
}
```

Here `print_caller()` appears to take no arguments, but we compile it to something like this:

```rust
#![feature(panic_internals)]
use std::panic::Location;

fn print_caller(caller: &Location) {
    println!("called from {}", caller);
}

fn main() {
    print_caller(&Location::internal_constructor(file!(), line!(), column!()));
}
```

### Dynamic dispatch

In codegen contexts we have to modify the callee ABI to pass this information down the stack, but
the attribute expressly does *not* modify the type of the function. The ABI change must be
transparent to type checking and remain sound in all uses.

Direct calls to tracked functions will always know the full codegen flags for the callee and can
generate appropriate code. Indirect callers won't have this information and it's not encoded in
the type of the function pointer they call, so we generate a [`ReifyShim`] around the function
whenever taking a pointer to it. This shim isn't able to report the actual location of the indirect
call (the function's definition site is reported instead), but it prevents miscompilation and is
probably the best we can do without modifying fully-stabilized type signatures.

> *Note:* We always emit a [`ReifyShim`] when taking a pointer to a tracked function. While the
> constraint here is imposed by codegen contexts, we don't know during MIR construction of the shim
> whether we'll be called in a const context (safe to ignore shim) or in a codegen context (unsafe
> to ignore shim). Even if we did know, the results from const and codegen contexts must agree.

## The attribute

The `#[track_caller]` attribute is checked alongside other codegen attributes to ensure the
function:

* has the `"Rust"` ABI (as opposed to e.g., `"C"`)
* is not a closure
* is not `#[naked]`

If the use is valid, we set [`CodegenFnAttrsFlags::TRACK_CALLER`][attrs-flags]. This flag influences
the return value of [`InstanceKind::requires_caller_location`][requires-location] which is in turn
used in both const and codegen contexts to ensure correct propagation.

### Traits

When applied to trait method implementations, the attribute works as it does for regular functions.

When applied to a trait method prototype, the attribute applies to all implementations of the
method. When applied to a default trait method implementation, the attribute takes effect on
that implementation *and* any overrides.

Examples:

```rust
#![feature(track_caller)]

macro_rules! assert_tracked {
    () => {{
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_ne!(location.line(), line!(), "line should be outside this fn");
        println!("called at {}", location);
    }};
}

trait TrackedFourWays {
    /// All implementations inherit `#[track_caller]`.
    #[track_caller]
    fn blanket_tracked();

    /// Implementors can annotate themselves.
    fn local_tracked();

    /// This implementation is tracked (overrides are too).
    #[track_caller]
    fn default_tracked() {
        assert_tracked!();
    }

    /// Overrides of this implementation are tracked (it is too).
    #[track_caller]
    fn default_tracked_to_override() {
        assert_tracked!();
    }
}

/// This impl uses the default impl for `default_tracked` and provides its own for
/// `default_tracked_to_override`.
impl TrackedFourWays for () {
    fn blanket_tracked() {
        assert_tracked!();
    }

    #[track_caller]
    fn local_tracked() {
        assert_tracked!();
    }

    fn default_tracked_to_override() {
        assert_tracked!();
    }
}

fn main() {
    <() as TrackedFourWays>::blanket_tracked();
    <() as TrackedFourWays>::default_tracked();
    <() as TrackedFourWays>::default_tracked_to_override();
    <() as TrackedFourWays>::local_tracked();
}
```

## Background/History

Broadly speaking, this feature's goal is to improve common Rust error messages without breaking
stability guarantees, requiring modifications to end-user source, relying on platform-specific
debug-info, or preventing user-defined types from having the same error-reporting benefits.

Improving the output of these panics has been a goal of proposals since at least mid-2016 (see
[non-viable alternatives] in the approved RFC for details). It took two more years until RFC 2091
was approved, much of its [rationale] for this feature's design having been discovered through the
discussion around several earlier proposals.

The design in the original RFC limited itself to implementations that could be done inside the
compiler at the time without significant refactoring. However in the year and a half between the
approval of the RFC and the actual implementation work, a [revised design] was proposed and written
up on the tracking issue. During the course of implementing that, it was also discovered that an
implementation was possible without modifying the number of arguments in a function's MIR, which
would simplify later stages and unlock use in traits.

Because the RFC's implementation strategy could not readily support traits, the semantics were not
originally specified. They have since been implemented following the path which seemed most correct
to the author and reviewers.

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
[attr-reference]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-track_caller-attribute
[intrinsic]: https://doc.rust-lang.org/nightly/core/intrinsics/fn.caller_location.html
[wrapper]: https://doc.rust-lang.org/nightly/core/panic/struct.Location.html#method.caller
[non-viable alternatives]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md#non-viable-alternatives
[rationale]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md#rationale
[revised design]: https://github.com/rust-lang/rust/issues/47809#issuecomment-443538059
[attrs-flags]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/middle/codegen_fn_attrs/struct.CodegenFnAttrFlags.html#associatedconstant.TRACK_CALLER
[`ReifyShim`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.InstanceKind.html#variant.ReifyShim
[`Location`]: https://doc.rust-lang.org/core/panic/struct.Location.html
[const-find-closest]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.InterpCx.html#method.find_closest_untracked_caller_location
[requires-location]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/instance/enum.InstanceKind.html#method.requires_caller_location
[alloc-location]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.InterpCx.html#method.alloc_caller_location
[fcx-location]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/struct.FunctionCx.html#structfield.caller_location
[const-location-query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.const_caller_location
[location-memory-kind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/enum.MemoryKind.html#variant.CallerLocation
[const-frame]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.Frame.html
[const-stack]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.InterpCx.html#structfield.stack
[fcx-get]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/struct.FunctionCx.html#method.get_caller_location
[frame-instance]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.Frame.html#structfield.instance
