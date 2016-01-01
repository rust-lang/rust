- Feature Name: `drop_types_in_const`
- Start Date: 2016-01-01
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow types with destructors to be used in `static` items and in `const` functions, as long as the destructor never needs to run in const context.

# Motivation
[motivation]: #motivation

Most collection types do not allocate any memory when constructed empty. With the change to make leaking safe, the restriction on `static` items with destructors
is no longer required to be a hard error.

Allowing types with destructors to be directly used in `const` functions and stored in `static`s will remove the need to have
runtime-initialisation for global variables.

# Detailed design
[design]: #detailed-design

- Lift the restriction on types with destructors being used in statics.
 - (Optionally adding a lint that warn about the possibility of resource leak)
- Alloc instantiating structures with destructors in constant expressions,
- Continue to prevent `const` items from holding types with destructors.
- Allow `const fn` to return types with destructors.
- Disallow constant expressions which would result in the destructor being called (if the code were run at runtime).

## Examples
Assuming that `RwLock` and `Vec` have `const fn new` methods, the following example is possible and avoids runtime validity checks.

```rust
/// Logging output handler
trait LogHandler: Send + Sync {
    // ...
}
/// List of registered logging handlers
static S_LOGGERS: RwLock<Vec< Box<LogHandler> >> = RwLock::new( Vec::new() );
```

Disallowed code
```rust
static VAL: usize = (Vec::<u8>::new(), 0).1;	// The `Vec` would be dropped
const EMPTY_BYTE_VEC: Vec<u8> = Vec::new();	// `const` items can't have destructors

const fn sample(_v: Vec<u8>) -> usize {
	0	// Discards the input vector, dropping it
}
```

# Drawbacks
[drawbacks]: #drawbacks

Destructors do not run on `static` items (by design), so this can lead to unexpected behavior when a type's destructor has effects outside the program (e.g. a RAII temporary folder handle, which deletes the folder on drop). However, this can already happen using the `lazy_static` crate.

# Alternatives
[alternatives]: #alternatives

- Runtime initialisation of a raw pointer can be used instead (as the `lazy_static` crate currently does on stable)
- On nightly, a bug related to `static` and `UnsafeCell<Option<T>>` can be used to remove the dynamic allocation.

Both of these alternatives require runtime initialisation, and incur a checking overhead on subsequent accesses.

# Unresolved questions
[unresolved]: #unresolved-questions

- TBD
