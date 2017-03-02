- Start Date: 2014-11-10
- RFC PR: https://github.com/rust-lang/rfcs/pull/458
- Rust Issue: https://github.com/rust-lang/rust/issues/22251

# Summary

I propose altering the `Send` trait as proposed by RFC #17 as
follows:

*   Remove the implicit `'static` bound from `Send`.
*   Make `&T` `Send` if and only if `T` is `Sync`.
    ```rust
    impl<'a, T> !Send for &'a T {}

    unsafe impl<'a, T> Send for &'a T where T: Sync + 'a {}
    ```
*   Evaluate each `Send` bound currently in `libstd` and either leave it as-is, add an
    explicit `'static` bound, or bound it with another lifetime parameter.

# Motivation

Currently, Rust has two types that deal with concurrency: `Sync` and `Send`

If `T` is `Sync`, then `&T` is threadsafe (that is, can cross task boundaries without
data races).  This is always true of any type with simple inherited mutability, and it is also true
of types with interior mutability that perform explicit synchronization (e.g. `Mutex` and
`Arc`).  By fiat, in safe code all static items require a `Sync` bound.  `Sync` is most
interesting as the proposed bound for closures in a fork-join concurrency model, where the thread
running the closure can be guaranteed to terminate before some lifetime `'a`, and as one of the
required bounds for `Arc`.

If `T` is `Send`, then `T` is threadsafe to send between tasks.  At an initial glance,
this type is harder to define.  `Send` currently requires a `'static` bound, which excludes
types with non-'static references, and there are a few types (notably, `Rc` and
`local_data::Ref`) that opt out of `Send`.  All static items other than those that are
`Sync` but not `Send` (in the stdlib this is just `local_data::Ref` and its derivatives)
are `Send`.  `Send` is most interesting as a required bound for `Mutex`, channels, `spawn()`, and
other concurrent types and functions.

This RFC is mostly motivated by the challenges of writing a safe interface for fork-join concurrency
in current Rust.  Specifically:

*   It is not clear what it means for a type to be `Sync` but not `Send`.  Currently there
    is nothing in the type system preventing these types from being instantiated.  In a fork-join
    model with a bounded, non-`'static` lifetime `'a` for worker tasks, using a
    `Sync + 'a` bound on a closure is the intended way to make sure the operation is safe to run
    in another thread in parallel with the main thread.  But there is no way of preventing the main
    and worker tasks from concurrently accessing an item that is `Sync + NoSend`.
*   Because `Send` has a `'static` bound, most concurrency constructs cannot be used if they have any non-static references in them, even in a thread with a bounded lifetime.  It seems like there should be a way to extend `Send` to shorter lifetimes.  But
    naively removing the `'static` bound causes memory unsafety in existing APIs like Mutex.

# Detailed Design

## Proposal

Extend the current meaning of `Send` in a (mostly) backwards-compatible way that
retains memory-safety, but allows for existing concurrent types like `Arc` and `Mutex` to be
used across non-`'static` boundaries.  Use `Send` with a bounded lifetime instead of `Sync` for fork-join concurrency.

The first proposed change is to remove the `'static` bound from `Send`.  Without doing this,
we would have to write brand new types for fork-join libraries that took `Sync` bounds but were
otherwise identical to the existing implementations.  For example, we cannot create a
`Mutex<Vec<&'a mut uint>>` as long as `Mutex` requires a `'static` bound.  By itself,
though, this causes unsafety.  For example, a `Mutex<&'a Cell<bool>>` does not necessarily
actually lock the data in the `Cell`:

```rust
let cell = Cell:new(true);
let ref_ = &cell;
let mutex = Mutex::new(&cell);
ref_.set(false); // Modifying the cell without locking the Mutex.
```

This leads us to our second refinement.  We add the rule that `&T` is `Send` if and only if
`T` is `Sync`--in other words, we disallow `Send`ing shared references with a
non-threadsafe interior.  We do, however, still allow `&mut T` where `T` is `Send`, even
if it is not `Sync`.  This is safe because `&mut T` linearizes access--the only way to
access the the original data is through the unique reference, so it is safe to send to other
threads.  Similarly, we allow `&T` where `T` is `Sync`, even if it is not `Send`, since by the definition of `Sync` `&T` is already known to be threadsafe.

Note that this definition of `Send` is identical to the old definition of `Send` when
restricted to `'static` lifetimes in safe code.  Since `static mut` items are not accessible
in safe code, and it is not possible to create a safe `&'static mut` outside of such an item, we
know that if `T: Send + 'static`, it either has only `&'static` references, or has no references at
all.  Since `'static` references can only be created in `static` items and literals in safe code, and
all `static` items (and literals) are `Sync`, we know that any such references are `Sync`.  Thus, our
new rule that `T` must be `Sync` for `&'static T` to be `Send` does not actually
remove `Send` from any existing types.  And since `T` has no `&'static mut` references,
unless any were created in unsafe code, we also know that our rule allowing `&'static mut T`
did not add `Send` to any new types.  We conclude that the second refinement is backwards compatible
with the old behavior, provided that old interfaces are updated to require `'static` bounds and they did not
create unsafe `'static` and `'static mut` references.  But unsafe types like these were already not
guaranteed to be threadsafe by Rust's type system.

Another important note is that with this definition, `Send` will fulfill the proposed role of `Sync` in a fork-join concurrency library.  At present, to use `Sync` in a fork-join library one must make the implicit assumption that if `T` is `Sync`, `T` is `Send`.  One might be tempted to codify this by making `Sync` a subtype of `Send`.  Unfortunately, this is not always the case, though it should be most of the time.  A type can be created with `&mut` methods that are not thread safe, but no `&`-methods that are not thread safe.  An example would be a version of `Rc` called `RcMut`.  `RcMut` would have a `clone_mut()` method that took `&mut self` and no other `clone()` method.  `RcMut` could be thread-safely shared provided that a `&mut RcMut` was not sent to another thread.  As long as that invariant was upheld, `RcMut` could only be cloned in its original thread and could not be dropped while shared (hence, `RcMut` is `Sync`) but a mutable reference could not be thread-safely shared, nor could it be moved into another thread (hence, `&mut RcMut` is not `Send`, which means that `RcMut` is not `Send`).  Because `&T` is Send if `T` is Sync (per the new definition), adding a `Send` bound will guarantee that only shared pointers of this type are moved between threads, so our new definition of `Send` preserves thread safety in the presence of such types.

Finally, we'd hunt through existing instances of `Send` in Rust libraries and replace them with
sensible defaults.  For example, the `spawn()` APIs should all have `'static` bounds,
preserving current behavior.  I don't think this would be too difficult, but it may be that there
are some edge cases here where it's tricky to determine what the right solution is.

## More unusual types

We discussed whether a type with a destructor that manipulated thread-local data could be non-`Send` even though `&mut T` was.  In general it could not, because you can call a destructor through `&mut` references (through `swap` or simply assigning a new value to `*x` where `x: &mut T`).  It was noted that since `&uniq T` cannot be dropped, this suggests a role for such types.

Some unusual types proposed by `arielb1` and myself to explain why `T: Send` does not mean `&mut T` is threadsafe, and `T: Sync` does not imply `T: Send`.  The first type is a bottom type, the second takes `self` by value (so `RcMainTask` is not `Send` but `&mut RcMainTask` is `Send`).

Comments from arielb1:

Observe that `RcMainTask::main_clone` would be unsafe outside the main task.

`&mut Xyz` and `&mut RcMainTask` are perfectly fine `Send` types. However, `Xyz` is a bottom (can be used to violate memory safety), and `RcMainTask` is not `Send`.

```rust
#![feature(tuple_indexing)]
use std::rc::Rc;
use std::mem;
use std::kinds::marker;

// Invariant: &mut Xyz always points to a valid C xyz.
// Xyz rvalues don't exist.

// These leak. I *could* wrap a box or arena, but that would
// complicate things.

extern "C" {
    // struct Xyz;
    fn xyz_create() -> *mut Xyz;
    fn xyz_play(s: *mut Xyz);
}

pub struct Xyz(marker::NoCopy);

impl Xyz {
    pub fn new() -> &'static mut Xyz {
        unsafe {
            let x = xyz_create();
            mem::transmute(x)
        }
    }

    pub fn play(&mut self) {
        unsafe { xyz_play(mem::transmute(self)) }
    }
}

// Invariant: only the main task has RcMainTask values

pub struct RcMainTask<T>(Rc<T>);
impl<T> RcMainTask<T> {
    pub fn new(t: T) -> Option<RcMainTask<T>> {
        if on_main_task() {
            Some(RcMainTask(Rc::new(t)))
        } else { None }
    }

    pub fn main_clone(self) -> (RcMainTask<T>, RcMainTask<T>) {
        let new = RcMainTask(self.0.clone());
        (self, new)
    }
}

impl<T> Deref<T> for RcMainTask<T> {
    fn deref(&self) -> &T { &*self.0 }
}

//  - by Sharp

pub struct RcMut<T>(Rc<T>);
impl<T> RcMut<T> {
    pub fn new(t: T) -> RcMut<T> {
        RcMut(Rc::new(t))
    }

    pub fn mut_clone(&mut self) -> RcMut<T> {
        RcMut(self.0.clone())
    }
}

impl<T> Deref<T> for RcMut<T> {
    fn deref(&self) -> &T { &*self.0 }
}

// fn on_main_task() -> bool { false /* XXX: implement */ }
// fn main() {}
```

# Drawbacks

Libraries get a bit more complicated to write, since you may have to write `Send + 'static` where previously you just wrote `Send`.

# Alternatives

We could accept the status quo.  This would mean that any existing `Sync` `NoSend`
type like those described above would be unsafe (that is, it would not be possible to write a non-`'static` closure with the correct bounds to make it safe to use), and it would not be possible to write a type like `Arc<T>` for a `T` with a bounded lifetime, as well as other safe concurrency constructs for fork-join concurrency.  I do not think this is a good alternative.

We could do as proposed above, but change `Sync` to be a subtype of `Send`.  Things wouldn't be too
different, but you wouldn't be able to write types like those discussed above.  I am not sure that types like that are actually useful, but even if we did this I think you would usually want to use a `Send` bound anyway.

We could do as proposed above, but instead of changing `Send`, create a new type for this
purpose.  I suppose the advantage of this would be that user code currently using `Send` as a way to
get a `'static` bound would not break.  However, I don't think it makes a lot of sense to keep the
current `Send` type around if this is implemented, since the new type should be backwards compatible
with it where it was being used semantically correctly.

# Unresolved questions

*   Is the new scheme actually safe?  I *think* it is, but I certainly haven't proved it.

*   Can this wait until after Rust 1.0, if implemented?  I think it is backwards incompatible, but I
believe it will also be much easier to implement once opt-in kinds are fully implemented.

*   Is this actually necessary?  I've asserted that I think it's important to be able to do the same
things in bounded-lifetime threads that you can in regular threads, but it may be that it isn't.

*   Are types that are `Sync` and `NoSend` actually useful?
