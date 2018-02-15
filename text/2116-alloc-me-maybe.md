- Feature Name: fallible_collection_alloc
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [rust-lang/rfcs#2116](https://github.com/rust-lang/rfcs/pull/2116)
- Rust Issue: [rust-lang/rust#48043](https://github.com/rust-lang/rust/issues/48043)

# Summary
[summary]: #summary

Add minimal support for fallible allocations to the standard collection APIs. This is done in two ways:

* For users with unwinding, an `oom=panic` configuration is added to make global allocators panic on oom.
* For users without unwinding, a `try_reserve() -> Result<(), CollectionAllocErr>` method is added.

The former is sufficient to unwinding users, but the latter is insufficient for the others (although it is a decent 80/20 solution). Completing the no-unwinding story is left for future work.


# Motivation
[motivation]: #motivation

Many collection methods may decide to allocate (push, insert, extend, entry, reserve, with_capacity, ...) and those allocations may fail. Early on in Rust's history we made a policy decision not to expose this fact at the API level, preferring to abort. This is because most developers aren't prepared to handle it, or interested. Handling allocation failure haphazardly is likely to lead to many never-tested code paths and therefore bugs. We call this approach *infallible* collection allocation, because the developer model is that allocations just don't fail.

Unfortunately, this stance is unsustainable in several of the contexts Rust is designed for.
This RFC seeks to establish a basic *fallible* collection allocation API, which allows our users to handle allocation failures where desirable. This RFC does not attempt to perfectly address all use cases, but does intend to establish the goals and constraints of those use cases, and sketches a path forward for addressing them all.

There are 4 user profiles we will be considering in this RFC:

* embedded: task-oriented, robust, pool-based, no unwinding
* gecko: semi-task-oriented, best-effort, global, no unwinding
* server: task-oriented, semi-robust, global, unwinding
* runtime: whole-system, robust, global, no unwinding




## User Profile: Embedded

Embedded devs are primarily well-aligned with Rust's current strategy. First and foremost, embedded devs just try to *not* dynamically allocate. Memory should ideally all be allocated at startup. In cases where this isn't practical, simply aborting the process is often the next-best choice. Robust embedded systems need to be able to recover from a crash anyway, and aborting is completely fool-proof.

However, sometimes the embedded system needs to process some user-defined tasks with unpredictable allocations, and completely crashing on OOM would be inappropriate. In those cases handling allocation failure is the right solution. In the case of a failure, the entire task usually reports a failure and is torn down. To make this robust, all allocations for a task are usually isolated to a single pool that can easily be torn down. This ensures nothing leaks, and helps avoid fragmentation. The important thing to note is that the embedded developers are ready and willing to take control of all allocations to do this properly.

Some embedded systems do use unwinding, but this is very rare, so it cannot be assumed.

It seems they would be happy to have some system to prevent infallible allocations from ever being used.




## User Profile: Gecko

Gecko is also primarily well-aligned with Rust's current strategy. For the most part, they liberally allocate and are happy to crash on OOM. This is especially palatable now that firefox is multiprocess. However as a *quality of implementation* matter, they occasionally make some subroutines fallible. For instance, it would be unfortunate if a single giant image prevented a page from loading. Similarly, running out of memory while processing a style sheet isn't significantly different from failing to download it.

However in contrast to the embedded case, this isn't done in a particularly principled way. Some parts might be fallible, some might be infallible. Nothing is pooled to isolate tasks. It's just a best-effort affair.

Gecko is built without unwinding.

It seems they would be happy to have some system to prevent infallible allocations from ever being used.

Gecko's need for this API as soon as possible will result in it temporarily forking several of the std collections, which is the primary impetus for this RFC.





## User Profile: Server

This represents a commodity server which handles tasks using threads or futures.

Similar to the embedded case, handling allocation failure at the granularity of tasks is ideal for quality-of-implementation purposes. However, unlike embedded development, it isn't considered practical (in terms of cost) to properly take control of everything and ensure allocation failure is handled robustly.

Here unwinding is available, and seems to be the preferred solution, as it maximizes the chances of allocation failures bubbling out of whatever libraries are used. This is unlikely to be totally robust, but that's ok.

With unwinding there isn't any apparent use for an infallible allocation checker.





## User Profile: Runtime

A garbage-collected runtime (such as SpiderMonkey or the Microsoft CLR), is generally expected to avoid crashing due to out-of-memory conditions. Different strategies and allocators are used for different situations here. Most notably, there are allocations on the GC heap for the running script, and allocations on the global heap for the actual runtime's own processing (e.g. performing a JIT compilation).

Allocations on the GC heap aren't particularly interesting for our purposes, as these need to have a special format for tracing, and management by the runtime. A runtime probably wouldn't ever want to build a native Vec backed by the GC heap, but a Vec *might* contain GC'd pointers that the runtime must trace. Thankfully, this is unrelated to the process of allocating the Vec itself.

When performing a GC, allocating data structures may enable faster or more responsive strategies, but the system must be ready to fall back to less memory-intensive solution in the case of allocation failure. In the limit, very small allocations in critical sections may be infallible.

When performing a JIT, running out of memory can generally be gracefully handled by failing the compilation and remaining in a less-optimized mode (such as the interpreter). For the most part fallible allocation is used here. However SpiderMonkey occasionally uses an interesting mix of fallible and infallible allocations to avoid threading errors through some particularly complex subroutines. Essentially, a chunk of memory is reserved that is supposed to be statically guaranteed to be sufficient for the subroutine to complete its task, and all allocations in the subroutine are subsequently treated as infallible. In debug builds, running out of memory will trigger an abort. In release builds they will first try to just get more memory and proceed, but abort if this fails.

Although the language the runtime hosts may have an unwinding/exceptions for OOM conditions when the GC heap runs out of space, the runtime itself generally doesn't use unwinding to handle its own allocation failures.

Due to mixed fallible/infallible allocation use, tools which prevent the use of infallible allocation may not be appropriate.

The Runtime dev profile seems to closely reflect that of Database dev (which wasn't seriously researched for this RFC). A database is in some sense just a runtime for its query language (e.g. SQL), with similar reliability constraints.

Aside: many devs in this space have a testing feature which can repeatedly run test cases with OOMs injected at the allocator level. This doesn't really effect our constraints, but it's something to keep in mind to address the "many untested paths" issue.




## Additional Background: How Collections Handle Allocation Now

All of our collections consider there to be two interesting cases:

* The capacity got too big (>`isize::MAX`), which is handled by `panic!("capacity overflow")`
* The allocator returned an err (even Unsupported), which is handled by calling `allocator.oom()`

To make matters more complex, on 64-bit platforms we don't check the `isize::MAX` condition directly, instead relying on the allocator to deterministically fail on any request that far exceeds a quantity the page table can even support (no 64-bit system we support uses all 64 bits of the pointer, even with new-fangled 5-level page tables). This means that 64-bit platforms behave slightly different on catastrophically large allocations (abort instead of panic).

These behaviours were purposefully designed, but probably not particularly well-motivated, [as discussed here](https://github.com/rust-lang/rust/issues/42808). Some of these details are documented, although not correctly or in sufficient detail. For instance `Vec::reserve` only mentions panicking when overflowing `usize`, which is accurate for 64-bit but not 32-bit or 16-bit. Oddly no mention of out-of-memory conditions or aborts can be found anywhere in Vec's documentation.

To make matters more complex, the (unstable) `heap::Alloc` trait currently documents that any oom impl can panic *or* abort, so collection users need to assume that can happen anyway. This is intended insofar as it was considered desirable for local allocators, but is considered an oversight in the global case. This is because Alloc is mostly designed around local allocators.

This is enough of a mess (which to be clear can be significantly blamed on the author) that the author expects no one is relying on the specific behaviours here, and they could be changed pretty liberally. That said, the primary version of this proposal doesn't attempt to change any of these behaviours. It's certainly a plausible alternative, though.




## Additional Background: Allocation Failure in C(++)

There are two ways that collection allocation failure is handled in C(++): with error return values, and with unwinding (C++ only). The C++ standard library (STL) only provides fallible allocations through exceptions, but the broader ecosystem also uses return values. For example, mozilla's own standard library (MFBT) only uses return values.

Unfortunately, attempting to handle allocation failure in C(++) has been a historical source of critical vulnerabilities. For instance, if reallocating an array fails but isn't noticed, the user of the array can end up thinking it has more space than it actually does and writing past the end of the allocation.

The return-value-based approach is problematic because neither language has good facilities for mandating that a result is actually *checked*. There are two notable cases here: when the result of the allocation is some kind of error code (e.g. a bool), or the result is a pointer into the allocation (or a specific pointer indicating failure).

In the error code case, neither language provides a native facility to mandate that error codes must be checked. However compiler-specific attributes like GCC's warn_unused_result can be used here. Unfortunately nothing mandates that the error code is used *correctly*. In the pointer case, blindly dereferencing is considered a valid use, fooling basic lints.

Unwinding is better than error codes in this regard, because completely ignoring an exception aborts the process. The author's understanding is that problems arise from the complicated exception-safety rules C++ collections have.

Both of these concerns are partially mitigated in Rust. For return values, Result and bool have proper on-by-default must-use checks. However again nothing mandates they are used properly. In the pointer case, we can however prevent you from ever getting the pointer if the Result is an `Err`. For unwinding, it's much harder to run afoul of exception-safety in Rust, especially since copy/move can't be overloaded. However unsafe code may have trouble.




## Additional Background: Overcommit and Killers

Some operating systems can be configured to pretend there's more memory than there actually is. Generally this is the result of pretending to allocate physical pages of memory, but only actually doing so when the page is accessed. For instance, forking a process is supposed to create two separate copies of the process's memory, but this can be avoided by simply marking all the pages as *copy on write* and having the processes share the same physical memory. The first process to mutate the shared page triggers a page fault, which the OS handles by properly allocating a new physical page for it. Similarly, to postpone zeroing fresh pages of memory, the OS may use a copy-on-write zero page.

The result of this is that allocation failure may happen when memory is first *accessed* and not when it's actually requested. If this happens, someone needs to give up their memory, which can mean the OS killing your process (or another random one!).

This strategy is used on many *nix variants/descendants, including Android, iOS, MacOS, and Ubuntu.

Some developers will try to use this as an argument for never *trying* to handle allocation failure. This RFC does not consider this to be a reasonable stance. First and foremost: Windows doesn't do it. So anything that's used a lot on windows (e.g. Firefox) can reasonably try to handle allocation failure there. Similarly, overcommit can be disabled completely or partially on many OSes. For instance the default for Linux is to actually fail on allocations that are "obviously" too large to handle.





## Additional Background: Recovering From Allocation Failure Without Data Loss

The most common collection interfaces in Rust expect you to move data into them, and may fail to allocate in the middle of processing this data. As a basic example, `push` consumes a T. To avoid data loss, this T should be returned, so a fallible `push` would need a signature like:

```rust
/// Inserts the given item at the end of the Vec.
///
/// If allocating space fails, the item is returned.
fn push(&mut self, item: T) -> Result<(), (T, Error)>;
```

More difficult is an API like `extend`, which in general cannot predict allocation size and so must continually reallocate while processing. It also cannot know if it needs space for an element until its been yielded by the iterator. As such extend might have a signature like:

```rust
/// Inserts all the items in the given iterator at the end of the Vec.
///
/// If allocating space fails, the collection will contain all the elements
/// that it managed to insert until the failure. The result will contain
/// the iterator, having been run up until the failure point. If the iterator
/// has been run at all, the last element yielded will also be returned.
fn extend<I: IntoIter<Item=T>>(&mut self, iter: I)
    -> Result<(), (I::IntoIter, Option<T>, Err)>
```

Note that this API only even works because Iterator's signature currently guarantees that the yielded elements outlive the iterator. This would not be the case if we ever moved to support so-called "streaming iterators", which yield elements that point into themselves.





# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Due to the diversity of requirements between our user profiles, there isn't any one-size fits all solution. This RFC proposes two solutions which will require minimal work for maximal impact:

* For the server users, an `oom=panic` configuration, in the same vein as the `panic=abort`.
* For everyone else, add `try_reserve` and `try_reserve_exact` as standard collection APIs.



## oom=panic

Applying this configuration in a Cargo.toml would change the behaviour of the global allocator's `oom()` function, which currently aborts, to instead panic. As discussed in the Server user profile, this would allow OOM to be handled at task boundaries with minimal effort for server developers, and no effort from library maintainers.

If using a thread-per-task model, OOMs will be naturally caught at the thread boundary. If using a different model, tasks can be isolated using the `thread::catch_unwind` or `Future::catch_unwind` APIs.

We expose a flag, rather than changing the default, because we maintain that *by default* Rust programmers should not be trying to recover from allocation failures.

For instance, a project which desires to work this way would add this to their Cargo.toml:

```toml
[profile]
oom = panic
```

And then in their application, do something like this:

```rust
fn main() {
    set_up_event_queue();
    loop {
        let event = get_next_event();
        let result = ::std::panic::catch_unwind(|| {
            process_event(&mut event)
        });

        if let Err(err) = result {
            if let Some(message) = err.downcast_ref::<&str>() {
                eprintln!("Task crashed: {}", message);
            } else if let Some(message) = err.downcast_ref::<String>() {
                eprintln!("Task crashed: {}", message);
            } else {
                eprintln!("Task crashed (unknown cause)");
            }

            // Handle failure...
        }
    }
}
```




## try_reserve

`try_reserve` and `try_reserve_exact` would be added to `HashMap`, `Vec`, `String`, and `VecDeque`. These would have the exact same APIs as their infallible counterparts, except that OOM would be exposed as an error case, rather than a call to `Alloc::oom()`. They would have the following signatures:

```
/// Tries to reserve capacity for at least `additional` more elements to be inserted
/// in the given `Vec<T>`. The collection may reserve more space to avoid
/// frequent reallocations. After calling `reserve`, capacity will be
/// greater than or equal to `self.len() + additional`. Does nothing if
/// capacity is already sufficient.
///
/// # Errors
///
/// If the capacity overflows, or the allocator reports a failure, then an error
/// is returned. The Vec is unmodified if this occurs.
pub fn try_reserve(&mut self, extra: usize) -> Result<(), CollectionAllocErr>;

/// Ditto, but has reserve_exact's behaviour
pub fn try_reserve_exact(&mut self, extra: usize) -> Result<(), CollectionAllocErr>;

/// Augments `AllocErr` with a CapacityOverflow variant.
pub enum CollectionAllocErr {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,
    /// Error due to the allocator (see the `AllocErr` type's docs).
    AllocErr(AllocErr),
}
```

We propose only these methods because they represent a minimal building block that third parties can develop fallible allocation APIs on top of. For instance, here are some basic implementations:

```
impl<T> FallibleVecExt<T> for Vec<T> {
    fn try_push(&mut self, val: T) -> Result<(), (T, Err)> {
        if let Err(err) = self.try_reserve(1) { return Err((val, err)) }
        self.push(val);
    }

    fn try_extend_exact<I>(&mut self, iter: T) -> Result<(), (I::IntoIter, Err)>
        where I: IntoIter,
              I::IntoIter: ExactSizeIterator<Item=T>, // note this!
    {
        let iter = iter.into_iter();

        if let Err(err) = self.try_reserve(iter.len()) { return Err((iter, err)) }

        self.extend(iter);
    }
}
```

Note that iterator-consuming implementations are limited to ExactSizeIterator, as this lets us perfectly predict how much space we need. In practice this shouldn't be much of a constraint, as most uses of these APIs just feed arrays into arrays or maps into maps. Only things like `filter` produce unpredictable iterator sizes.





# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation



## oom=panic

Disclaimer: not super familiar with all the mechanics here, so this is a sketch that hopefully someone whose worked on these details can help flesh out.

We add a `-C oom=abort|panic` flag to rustc, which changes the impl of `__rust_oom` that's linked in to either panic or abort. It's possible that this should just change the value of a `extern static bool` in libcore (liballoc?) that `__rust_oom` impls are expected to check?

Unlike the `panic=abort` flag, this shouldn't make your crate incompatible with crates with a different choice. Only a subset of target types should be able to set this, e.g. it's a bin-level decision?

Cargo would also add a `oom=abort=panic` profile configuration, to set the rustc flag. Its value should be ignored in dependencies?



## try_reserve

[An implementation of try_reserve for Vec can be found for here](https://github.com/rust-lang/rust/pull/43890)

The guide-level explanation otherwise covers all the interesting details.




# Drawbacks
[drawbacks]: #drawbacks

There doesn't seem to be any drawback for adding support for `oom=panic`.

`try_reserve`'s only serious drawback is that it isn't a complete solution, and it may not idiomatically match future "complete" solutions to the problem.





# Rationale and Alternatives
[alternatives]: #alternatives



## Always panic on OOM

We probably shouldn't mandate this in the actual Alloc trait, but certainly we could change how our global Alloc impls behave. This RFC doesn't propose this for two reasons.

The first is basically on the grounds of "not rocking the boat". Notably unsafe code might be relying on global OOM not unwinding for exception safety reasons. The author expects such code could very easily be changed to be exception-safe if we decided to do this.

The second is that the author still considers it legitimately correct to discourage handling OOM by default, for the reasons stated in earlier sections.




## Eliminate the CapacityOverflow distinction

Collections could potentially just create an `AllocErr::Unsupported("capacity overflow")` and feed it to their allocator. Presumably this wouldn't do something bad to the allocator? Then the oom=abort flag could be used to completely control whether allocation failure is a panic or abort (for participating allocators).

Again this is avoided simply to leave things "as they are". In this case it would be a change to a legitimately documented API behaviour (panic on overflow of usize), but again that documentation isn't even totally accurate.




## Eliminate the 64-bit difference

This difference literally exists to save a single perfectly-predictable compare-and-branch on 64-bit platforms when allocating collections, which is probably insignificant considering how expensive the success path is. Also the difference here would be a bit exacerbated by exposing the CapacityOverflow variant here.

Again, not proposed to avoid rocking the boat.




## CollectionAllocErr

There were a few different possible designs for CollectionAllocErr:

* Just make it AllocErr
* Remove the payload from the AllocErr variant
* Just make it a `()` (so try_reserve basically returns a bool)

AllocErr already has an `Unsupported(&'static str)` variant to capture any miscellaneous allocation problems, so CapacityOverflow could plausibly just be stuffed in there. We opted to keep it separate to most accurately reflect the way collections think about these problems today -- CapacityOverflow goes to panic and AllocErr goes to `oom()`. It's possible end users simply don't care, in much the same way that collections don't actually care if an AllocErr is `Exhausted` or `Unsupported`.

It's also possible we should suppress the AllocErr details to "hide" how collections are interpreting the requests they receive. This just didn't seem that important, and has the possibility to get in the way of someone using their own local allocator.

The most extreme version of this would be to just say "there was an error" without any information. The only reason to really prefer this is for bloat reasons; the current Rust compiler really doesn't handle Result payloads very efficiently. This should presumably be fixed *eventually*, since Results are pretty important?

We simply opted for the version that had maximum information, on the off-chance this was useful.




## Future Work: Infallible Allocation Effect System (w/ Portability Lints)

Several of our users have expressed desire for some kind of system to prevent a function from ever infallibly allocating. This is ultimately an effect system.

One possible way to implement this would be to use the *portability lint* system. In particular, the "subsetting" portability lints that were proposed as future work in [RFC-1868](https://github.com/rust-lang/rfcs/blob/master/text/1868-portability-lint.md#subsetting-std).

This system is supposed to handle things like "I don't have float support" or "I don't have AtomicU64". "I don't have infallible allocation support" is much the same idea. This could be scoped to modules or functions.


## Future Work: Complete Result APIs

Although this RFC handles the "wants to unwind" case pretty cleanly and completely, it leaves no-unwind world with an imperfect one. In particular, it's completely useless for collections which have unpredictable allocations like BTreeMap. This proposal punts on this problem because solving it will be a big change which will likely make a bunch of people mad no matter what.

The author would prefer that we don't spend much time focusing on these solutions, but will document them here just for informational purposes. Also for these purposes we will only be discussing the `push` method on Vec, since any solution for that generalizes cleanly to everything else.

Broadly speaking, there's two schools of thought here: fallible operations should just be methods, and fallible operations should be distinguished at the type-level. Basically, should you be able to do: `vec.push(x); vec.try_push(y)`, or will you somehow obtain a special kind of Vec and `vec.push(x)` will then return a `Result`.

It should be noted that this appears to be a source of massive disagreement. Even within the gecko codebase, there are supporters of both approaches, and so it actually supports both. This is probably not a situation we should strive to emulate.

There are a few motivations for a type-level distinction:

* If it's done through a default generic parameter, then code can be written generically over doing something fallibly or infallibly
* If it's done through a default generic parameter, it potentially enables code reuse in implementations
* It can allow you to enforce that all operations on a Vec are performed fallibly
* It can make usage more ergonomic (no need for `try_` in front of everything)

The first doesn't appear to actually do much semantically. Code that's generic over fallibility is literally the exact same as code that only uses the fallible APIs, at which point you might as well just toss an `expect` at the end if you want to crash on OOM. The only difference seems to be the performance difference between propagating Results vs immediately unwinding/aborting. This can certainly be significant in code that's doing a lot of allocations, but it's not really clear how much this matters. Especially if Result-based codegen improves (which there's a lot of room for).

The second is interesting, but mostly effects collection implementors. Making users deal with additional generic parameters to make implementations easier doesn't seem very compelling.

Also these two benefits must be weighed against the cost of default generic parameters: they don't work very well (and may never?), and most people won't bother to support them so using a non-default just makes you incompatible with a bunch of the ecosystem.

The third is a bit more compelling, but has a few issues. First, it doesn't actually enforce that a function handles all allocation failures. One can create a fresh Vec, Box, or just call into a routine that allocates like `slice::sort()` and types won't do anything to prevent this. Second, it's a fairly common pattern to fallibly reserve space, and then infallibly insert data. For instance, code like the following can be found in many places in Gecko's codebase:

```rust
fn process(&mut self, data: &[Item]) -> Result<Vec<Processed>, CollectionAllocErr> {
    let mut vec = FallibleVec::new();
    vec.reserve(data.len())?

    for x in data {
        let p = process(x);
        self.push(p).unwrap();  // Wait, is this fallible or not?
    }
}
```

Mandating all operations be fallible can be confusing in that case (and has similar inefficiencies to the ones discussed in the previous point). Although admittedly this is a lot better in Rust with must-be-unwrapped-Results. In Gecko, "unwrapping" is often just blindly dereferencing a pointer, which is Undefined Behaviour if the allocation actually fails.

The fourth is certainly nice-to-have, but probably not a high enough priority to create an entire separate Vec type.

All of the type-based solutions also suffer from a fairly serious problem: they can't implement many core traits in the fallible state. For instance, Extend::extend and Display::to_string require allocation and don't support fallibility.

With all that said, these are the proposed solutions:


### Method-Based

Fairly straight-forward, but a bunch of duplicate code. Probably we would either end up implementing `push` in terms of `try_push` (which would be inefficient but easy), or with macros.

```rust
impl<T> Vec<T> {
    fn try_push(&mut self, elem: T) -> Result<(), (T, CollectionAllocErr)> {
        if self.len() == self.capacity() {
            if let Err(e) = self.try_reserve(1) {
                return Err((elem, e));
            }
        }

        // ... do actual push normally ...
    }
}
```


### Generic (on Vec)

This is a sketch, didn't want to put enough effort in to crack this puzzle.

The most notable thing is that it relies on generic associated types, which
don't actually exist yet, and probably won't be stable until ~late 2018
(optimistically).

```rust
trait Fallibility {
    type Result<T, E>;
    fn ok<T, E>(val: T) -> Self::Result<T, E>;
    fn err<T, E>(val: E, details: CollectionAllocErr) -> Self::Result<T, E>;
    // ... probably some other stuff here...?
}

struct Fallible;
struct Infallible;

impl Fallibility for Fallible {
    type Result<T, E> = Result<T, (E, CollectionAllocErr)>;
    fn ok<T, E>(val: T) -> Self::Result<T, E> {
        Ok(val)
    }
    fn err<T, E>(val: E, details: CollectionAllocErr) -> Self::Result<T, E> {
        Err((val, details))
    }
}

impl Fallibility for Infallible {
    type Result<T, E> = T;
    fn ok<T, E>(val: T) -> Self::Result<T, E> {
        val
    }
    fn err<T, E>(val: E, defaults: CollectionAllocErr) -> Self::Result<T, E> {
        unreachable!() // ??? maybe ???
    }
}

struct Vec<T, ..., F: Fallibility=Infallible> { ... }

impl<T, ..., F> Vec<T, ..., F> {
    fn push(&mut self) -> F::Result<(), T> {
        if self.len() == self.capacity() {
            let result = self.reserve(1);
            // ??? How do I match on this in generic code ???
            // (can't use Carrier since we need to add `elem` payload?)
            if result.is_err() {
                // Have to move elem into closure,
                // so can only map_err conditionally
                return result.map_err(move |err| (elem, err));
            }
        }

        // ... do actual push normally ...
    }
}
```



### Generic (on Alloc)

Same basic idea as the previous design, but the Fallibility trait is folded into the Alloc trait. Then one would use `FallibleHeap` or `InfallibleHeap`, or maybe `Infallible<Heap>`? This forces anyone who wants to support generic allocators to support generic fallibility. It would require a complete redesign of the allocator API, blocking it on generic associated types.



### FallibleVec

Just make a completely separate type. Includes an `into_fallible(self)`/`into_infallible(self)` conversion which is free since there's no actual representation change. Makes it possible to change "phases" between fallibility/infallibly for different parts of the program if that's valuable. Implementation-wise, basically identical to the method approach, but we also need to duplicate non-allocating methods just to mirror the API.

Alternatively we could make `FallibleVec<'a, T>` and `as_fallible(&mut self)`, which is a temporary view like Iterator/Entry. This is probably a bit more consistent with how we do this sort of thing. This also makes "temporary" fallibility easier, but at the cost of being able to permanently become fallible:

```rust
vec.as_fallible().push(x)?

// vs

let vec = vec.into_fallible();
vec.push(x)?
let vec = vec.into_infallible();

// but this actually works:

return vec.into_fallible()
```



# Unresolved questions
[unresolved]: #unresolved-questions

* How exactly should oom=panic be implemented in the compiler?
* How exactly should oom=panic behave for dependencies?
