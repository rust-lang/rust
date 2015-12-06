- Feature Name: allocator_api
- Start Date: 2015-12-01
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add a standard allocator interface and support for user-defined
allocators, with the following goals:

 1. Allow libraries (in libstd and elsewhere) to be generic with
    respect to the particular allocator, to support distinct,
    stateful, per-container allocators.

 2. Require clients to supply metadata (such as block size and
    alignment) at the allocation and deallocation sites, to ensure
    hot-paths are as efficient as possible.

 3. Provide high-level abstraction over the layout of an object in
    memory.

Regarding GC: We plan to allow future allocators to integrate
themselves with a standardized reflective GC interface, but leave
specification of such integration for a later RFC. (The design
describes a way to add such a feature in the future while ensuring
that clients do not accidentally opt-in and risk unsound behavior.)

# Motivation
[motivation]: #motivation

As noted in [RFC PR 39][] (and reiterated in [RFC PR 244][]), modern general purpose allocators are good,
but due to the design tradeoffs they must make, cannot be optimal in
all contexts.  (It is worthwhile to also read discussion of this claim
in papers such as
[Reconsidering Custom Malloc](#reconsidering-custom-memory-allocation).)

Therefore, the standard library should allow clients to plug in their
own allocator for managing memory.

## Allocators are used in C++ system programming

The typical reasons given for use of custom allocators in C++ are among the
following:

  1. Speed: A custom allocator can be tailored to the particular
     memory usage profiles of one client.  This can yield advantages
     such as:

     * A bump-pointer based allocator, when available, is faster
       than calling `malloc`.

     * Adding memory padding can reduce/eliminate false sharing of
       cache lines.

  2. Stability: By segregating different sub-allocators and imposing
     hard memory limits upon them, one has a better chance of handling
     out-of-memory conditions.

     If everything comes from a single global heap, it becomes much
     harder to handle out-of-memory conditions because by the time the
     handler runs, it is almost certainly going to be unable to
     allocate any memory for its own work.

  3. Instrumentation and debugging: One can swap in a custom
     allocator that collects data such as number of allocations,
     or time for requests to be serviced.

## Allocators should feel "rustic"

In addition, for Rust we want an allocator API design that leverages
the core type machinery and language idioms (e.g. using `Result`, with
a `NonZero` okay variant and a zero-sized error variant), and provides
premade functions for common patterns for allocator clients (such as
allocating either single instances of a type, or arrays of some types
of dynamically-determined length).

## Garbage Collection integration

Finally, we want our allocator design to allow for a garbage
collection (GC) interface to be added in the future.

At the very least, we do not want to accidentally *disallow* GC by
choosing an allocator API that is fundamentally incompatible with it.

(However, this RFC does not actually propose a concrete solution for
how to integrate allocators with GC.)

# Detailed design
[design]: #detailed-design

## The `Allocator` trait at a glance

The source code for the `Allocator` trait prototype ks provided in an
[appendix][Source for Allocator]. But since that section is long, here
we summarize the high-level points of the `Allocator` API.

(See also the [walk thru][] section, which actually links to
individual sections of code.)

 * Basic implementation of the trait requires just two methods
   (`alloc` and `dealloc`). You can get an initial implemention off
   the ground with relatively little effort.

 * All methods that can fail to satisfy a request return a `Result`
   (rather than building in an assumption that they panic or abort).
 
   * Furthermore, allocator implementations are discouraged from
     directly panicking or aborting on out-of-memory (OOM) during
     calls to allocation methods; instead,
     clients that do wish to report that OOM occurred via a particular
     allocator can do so via the `Allocator::oom()` method.

   * OOM is not the only type of error that may occur in general;
     allocators can inject more specific error types to indicate
     why an allocation failed.

 * The metadata for any allocation is captured in a `Kind`
   abstraction. This type carries (at minimum) the size and alignment
   requirements for a memory request.

   * The `Kind` type provides a large family of functional construction
     methods for building up the description of how memory is laid out.

     * Any sized type `T` can be mapped to its `Kind`, via `Kind::new::<T>()`,

     * Heterogenous structure; e.g. `kind1.extend(kind2)`,

     * Homogenous array types: `kind.repeat(n)` (for `n: usize`),

     * There are packed and unpacked variants for the latter two methods.

   * Helper `Allocator` methods like `fn alloc_one` and `fn
     alloc_array` allow client code to interact with an allocator
     without ever directly constructing a `Kind`.

 * Once an `Allocator` implementor has the `fn alloc` and `fn dealloc`
   methods working, it can provide overrides of the other methods,
   providing hooks that take advantage of specific details of how your
   allocator is working underneath the hood.

   * In particular, the interface provides a few ways to let clients
     potentially reuse excess memory associated with a block

   * `fn realloc` is a common pattern (where the client hopes that
     the method will reuse the original memory when satisfying the
     `realloc` request).

   * `fn alloc_excess` and `fn usable_size` provide an alternative
     pattern, where your allocator tells the client about the excess
     memory provided to satisfy a request, and the client can directly
     expand into that excess memory, without doing round-trip requests
     through the allocator itself.

## Semantics of allocators and their memory blocks
[semantics of allocators]: #semantics-of-allocators-and-their-memory-blocks

In general, an allocator provide access to a memory pool that owns
some amount of backing storage. The pool carves off chunks of that
storage and hands it out, via the allocator, as individual blocks of
memory to service client requests. (A "client" here is usually some
container library, like `Vec` or `HashMap`, that has been suitably
parameterized so that it has an `A:Allocator` type parameter.)

So, an interaction between a program, a collection library, and an
allocator might look like this:

<img width="800" src="https://rawgit.com/pnkfelix/pnkfelix.github.com/69230e5f1ea140c0a09c5a9fdd7f0766207cdddd/Svg/allocator-msc.svg">
If you cannot see the SVG linked here, try the [ASCII art version][ascii-art] appendix.
Also, if you have suggestions for changes to the SVG, feel free to write them as a comment
in that appendix; (but be sure to be clear that you are pointing out a suggestion for the SVG).
</img>

In general, an allocator might be the backing memory pool itself; or
an allocator might merely be a *handle* that references the memory
pool. In the former case, when the allocator goes out of scope or is
otherwise dropped, the memory pool is dropped as well; in the latter
case, dropping the allocator has no effect on the memory pool.

 * One allocator that acts as a handle is the global heap allocator,
   whose associated pool is the low-level `#[allocator]` crate.

 * Another allocator that acts as a handle is a `&'a Pool`, where
   `Pool` is some structure implementing a sharable backing store.
   The big [example][] section shows an instance of this.

 * An allocator that is its own memory pool would be a type
   analogous to `Pool` that implements the `Allocator` interface
   directly, rather than via `&'a Pool`.

 * A case in the middle of the two extremes might be something like an
   allocator of the form `Rc<RefCell<Pool>>`. This reflects *shared*
   ownership between a collection of allocators handles: dropping one
   handle will not drop the pool as long as at least one other handle
   remains, but dropping the last handle will drop the pool itself.

A client that is generic over all possible `A:Allocator` instances
cannot know which of the above cases it falls in. This has consequences
in terms of the restrictions that must be met by client code
interfacing with an allocator, which we discuss in a
later [section on lifetimes][lifetimes].


## Example Usage
[example]: #example-usage

Lets jump into a demo. Here is a (super-dumb) bump-allocator that uses
the `Allocator` trait.

### Implementing the `Allocator` trait

First, the bump-allocator definition itself: each such allocator will
have its own name (for error reports from OOM), start and limit
pointers (`ptr` and `end`, respectively) to the backing storage it is
allocating into, as well as the byte alignment (`align`) of that
storage, and an `avail: AtomicPtr<u8>` for the cursor tracking how
much we have allocated from the backing storage. 
(The `avail` field is an atomic because eventually we want to try
sharing this demo allocator across scoped threads.)

```rust
struct DumbBumpPool {
    name: &'static str,
    ptr: *mut u8,
    end: *mut u8,
    avail: AtomicPtr<u8>,
    align: usize,
}
```

The initial implementation is pretty straight forward: just immediately
allocate the whole pool's backing storage.

(If we wanted to be really clever we might layer this type on top of
*another* allocator.
For this demo I want to try to minimize cleverness, so we will use
`heap::allocate` to grab the backing storage instead of taking an
`Allocator` of our own.)


```rust
impl DumbBumpPool {
    fn new(name: &'static str,
           size_in_bytes: usize,
           start_align: usize) -> DumbBumpPool {
        unsafe {
            let ptr = heap::allocate(size_in_bytes, start_align);
            if ptr.is_null() { panic!("allocation failed."); }
            let end = ptr.offset(size_in_bytes as isize);
            DumbBumpPool {
                name: name,
                ptr: ptr, end: end, avail: AtomicPtr::new(ptr),
                align: start_align
            }
        }
    }
}
```

Since clients are not allowed to have blocks that outlive their
associated allocator (see the [lifetimes][] section),
it is sound for us to always drop the backing storage for an allocator
when the allocator itself is dropped
(regardless of what sequence of `alloc`/`dealloc` interactions occured
with the allocator's clients).

```rust
impl Drop for DumbBumpPool {
    fn drop(&mut self) {
        unsafe {
            let size = self.end as usize - self.ptr as usize;
            heap::deallocate(self.ptr, size, self.align);
        }
    }
}
```

Now, before we get into the trait implementation itself, here is an
interesting simple design choice:

 * To show-off the error abstraction in the API, we make a special
   error type that covers a third case that is not part of the
   standard `enum AllocErr`.

Specifically, our bump allocator has *three* error conditions that we
will expose:

 1. the inputs could be invalid,

 2. the memory could be exhausted, or,

 3. there could be *interference* between two threads.
    This latter scenario means that this allocator failed
    on this memory request, but the client might
    quite reasonably just *retry* the request.

```rust
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum BumpAllocError { Invalid, MemoryExhausted, Interference }

impl alloc::AllocError for BumpAllocError {
    fn invalid_input() -> Self { BumpAllocError::MemoryExhausted }
    fn is_memory_exhausted(&self) -> bool { *self == BumpAllocError::MemoryExhausted  }
    fn is_request_unsupported(&self) -> bool { false }
    fn is_transient(&self) { *self == BumpAllocError::Interference }
}
```

With that out of the way, here are some other design choices of note:

 * Our Bump Allocator is going to use a most simple-minded deallocation
   policy: calls to `fn dealloc` are no-ops. Instead, every request takes
   up fresh space in the backing storage, until the pool is exhausted.
   (This was one reason I use the word "Dumb" in its name.)

 * Since we want to be able to share the bump-allocator amongst multiple
   (lifetime-scoped) threads, we will implement the `Allocator` interface
   as a *handle* pointing to the pool; in this case, a simple reference.

Here is the demo implementation of `Allocator` for the type.

```rust
impl<'a> Allocator for &'a DumbBumpPool {
    type Kind = alloc::Kind;
    type Error = BumpAllocError;

    unsafe fn alloc(&mut self, kind: &Self::Kind) -> Result<Address, Self::Error> {
        let curr = self.avail.load(Ordering::Relaxed) as usize;
        let align = *kind.align();
        let curr_aligned = (curr.overflowing_add(align - 1)) & !(align - 1);
        let size = *kind.size();
        let remaining = (self.end as usize) - curr_aligned;
        if remaining <= size {
            return Err(BumpAllocError::MemoryExhausted);
        }

        let curr = curr as *mut u8;
        let curr_aligned = curr_aligned as *mut u8;
        let new_curr = curr_aligned.offset(size as isize);

        if curr != self.avail.compare_and_swap(curr, new_curr, Ordering::Relaxed) {
            return Err(BumpAllocError::Interference);
        } else {
            println!("alloc finis ok: 0x{:x} size: {}", curr_aligned as usize, size);
            return Ok(NonZero::new(curr_aligned));
        }
    }

    unsafe fn dealloc(&mut self, _ptr: Address, _kind: &Self::Kind) -> Result<(), Self::Error> {
        // this bump-allocator just no-op's on dealloc
        Ok(())
    }

    unsafe fn oom(&mut self) -> ! {
        panic!("exhausted memory in {}", self.name);
    }

}
```

(Niko Matsakis has pointed out that this particular allocator might
avoid interference errors by using fetch-and-add rather than
compare-and-swap. The devil's in the details as to how one might
accomplish that while still properly adjusting for alignment; in any
case, the overall point still holds in cases outside of this specific
demo.)

And that is it; we are done with our allocator implementation.

### Using an `A:Allocator` from the client side

We assume that `Vec` has been extended with a `new_in` method that
takes an allocator argument that it uses to satisfy its allocation
requests.

```rust
fn demo_alloc<A1:Allocator, A2:Allocator, F:Fn()>(a1:A1, a2: A2, print_state: F) {
    let mut v1 = Vec::new_in(a1);
    let mut v2 = Vec::new_in(a2);
    println!("demo_alloc, v1; {:?} v2: {:?}", v1, v2);
    for i in 0..10 {
        v1.push(i as u64 * 1000);
        v2.push(i as u8);
        v2.push(i as u8);
    }
    println!("demo_alloc, v1; {:?} v2: {:?}", v1, v2);
    print_state();
    for i in 10..100 {
        v1.push(i as u64 * 1000);
        v2.push(i as u8);
        v2.push(i as u8);
    }
    println!("demo_alloc, v1.len: {} v2.len: {}", v1.len(), v2.len());
    print_state();
    for i in 100..1000 {
        v1.push(i as u64 * 1000);
        v2.push(i as u8);
        v2.push(i as u8);
    }
    println!("demo_alloc, v1.len: {} v2.len: {}", v1.len(), v2.len());
    print_state();
}

fn main() {
    use std::thread::catch_panic;

    if let Err(panicked) = catch_panic(|| {
        let alloc = DumbBumpPool::new("demo-bump", 4096, 1);
        demo_alloc(&alloc, &alloc, || println!("alloc: {:?}", alloc));
    }) {
        match panicked.downcast_ref::<String>() {
            Some(msg) => {
                println!("DumbBumpPool panicked: {}", msg);
            }
            None => {
                println!("DumbBumpPool panicked");
            }
        }
    }

    // // The below will be (rightly) rejected by compiler when
    // // all pieces are properly in place: It is not valid to
    // // have the vector outlive the borrowed allocator it is
    // // referencing.
    //
    // let v = {
    //     let alloc = DumbBumpPool::new("demo2", 4096, 1);
    //     let mut v = Vec::new_in(&alloc);
    //     for i in 1..4 { v.push(i); }
    //     v
    // };

    let alloc = DumbBumpPool::new("demo-bump", 4096, 1);
    for i in 0..100 {
        let r = ::std::thread::scoped(|| {
            let v = Vec::new_in(&alloc);
            for j in 0..10 {
                v.push(j);
            }
        });
    }

    println!("got here");
}
```

And that's all to the demo, folks.

## Allocators and lifetimes
[lifetimes]: #allocators-and-lifetimes

As mentioned above, allocators provide access to a memory pool. An
allocator can *be* the pool (in the sense that the allocator owns the
backing storage that represents the memory blocks it hands out), or an
allocator can just be a handle that points at the pool.

Some pools have indefinite extent. An example of this is the global
heap allocator, requesting memory directly from the low-level
`#[allocator]` crate. Clients of an allocator with such a pool need
not think about how long the allocator lives; instead, they can just
freely allocate blocks, use them at will, and deallocate them at
arbitrary points in the future. Memory blocks that come from such a
pool will leak if it is not explicitly deallocated.

Other pools have limited extent: they are created, they build up
infrastructure to manage their blocks of memory, and at some point,
such pools are torn down. Memory blocks from such a pool may or may
not be returned to the operating system during that tearing down.

There is an immediate question for clients of an allocator with the
latter kind of pool (i.e. one of limited extent): whether it should
attempt to spend time deallocating such blocks, and if so, at what
time to do so?

Again, note:

 * generic clients (i.e. that accept any `A:Allocator`) *cannot know*
   what kind of pool they have, or how it relates to the allocator it
   is given,

 * dropping the client's allocator may or may not imply the dropping
   of the pool itself!

That is, code written to a specific `Allocator` implementation may be
able to make assumptions about the relationship between the memory
blocks and the allocator(s), but the generic code we expect the
standard library to provide cannot make such assumptions.

To satisfy the above scenarios in a sane, consistent, general fashion,
the `Allocator` trait assumes/requires all of the following:

 1. (for allocator impls and clients): in the absence of other
    information (e.g. specific allocator implementations), all blocks
    from a given pool have lifetime equivalent to the lifetime of the
    pool.

    This implies if a client is going to read from, write to, or
    otherwise manipulate a memory block, the client *must* do so before
    its associated pool is torn down.

    (It also implies the converse: if a client can prove that the pool
     for an allocator is still alive, then it can continue to work
     with a memory block from that allocator even after the allocator
     is dropped.)

 2. (for allocator impls): an allocator *must not* outlive its
    associated pool.

    All clients can assume this in their code.

    (This constraint provides generic clients the preconditions they
    need to satisfy the first condition. In particular, even though
    clients do not generally know what kind of pool is associated with
    its allocator, it can conservatively assume that all blocks will
    live at least as long as the allocator itself.)

 3. (for allocator impls and clients): all clients of an allocator
    *should* eventually call the `dealloc` method on every block they
    want freed (otherwise, memory may leak).

    However, allocator implementations *must* remain sound even if
    this condition is not met: If `dealloc` is not invoked for all
    blocks and this condition is somehow detected, then an allocator
    can panic (or otherwise signal failure), but that sole violation
    must not cause undefined behavior.

    (This constraint is to encourage generic client authors to write
     code that will not leak memory when instantiated with allocators
     of indefinite extent, such as the global heap allocator.)

 4. (for allocator impls): moving an allocator value *must not*
     invalidate its outstanding memory blocks.

     All clients can assume this in their code.

     So if a client allocates a block from an allocator (call it `a1`)
     and then `a1` moves to a new place (e.g. via`let a2 = a1;`), then
     it remains sound for the client to deallocate that block via
     `a2`.

     Note that this implies that it is not sound to implement an
     allocator that embeds its own pool structurally inline.

     E.g. this is *not* a legal allocator:
     ```rust
     struct MegaEmbedded { pool: [u8; 1024*1024], cursor: usize, ... }
     impl Allocator for MegaEmbedded { ... }
     ```
     The latter impl is simply unreasonable (at least if one is
     intending to satisfy requests by returning pointers into
     `self.bytes`).

     (Note of course, `impl Allocator for &mut MegaEmbedded` is in
     principle *fine*; that would then be an allocator that is an
     indirect handle to an unembedded pool.)

 5. (for allocator impls and clients) if an allocator is cloneable, the 
    client *can assume* that all clones
    are interchangably compatible in terms of their memory blocks: if
    allocator `a2` is a clone of `a1`, then one can allocate a block
    from `a1` and return it to `a2`, or vice versa, or use `a2.realloc`
    on the block, et cetera.

    This essentially means that any cloneable
    allocator *must* be a handle indirectly referencing a pool of some
    sort. (Though do remember that such handles can collectively share
    ownership of their pool, such as illustrated in the
    `Rc<RefCell<Pool>>` example given earlier.)

    (Note: one might be tempted to further conclude that this also
    implies that allocators implementing `Copy` must have pools of
    indefinite extent. While this seems reasonable for Rust as it
    stands today, I am slightly worried whether it would continue to
    hold e.g.  in a future version of Rust with something like
    `Gc<GcPool>: Copy`, where the `GcPool` and its blocks is reclaimed
    (via finalization) sometime after being determined to be globally
    unreachable. Then again, perhaps it would be better to simply say
    "we will not support that use case for the allocator API", so that
    clients would be able to employ the reasoning outlined in the
    outset of this paragraph.)


## A walk through the Allocator trait
[walk thru]: #a-walk-through-the-allocator-trait

### Role-Based Type Aliases

Allocation code often needs to deal with values that boil down to a
`usize` in the end. But there are distinct roles (e.g. "size",
"alignment") that such values play, and I decided those roles would be
worth hard-coding into the method signatures.

 * Therefore, I made [type aliases][] for `Size`, `Capacity`, `Alignment`, and `Address`.

Furthermore, all values of the above types must be non-zero for any
allocation action to make sense.

 * Therefore, I made them instances of the `NonZero` type.

### Basic implementation

An instance of an allocator has many methods, but an implementor of
the trait need only provide two method bodies: [alloc and dealloc][].

(This is only *somewhat* analogous to the `Iterator` trait in Rust. It
is currently very uncommon to override any methods of `Iterator` ecept
for `fn next`. However, I expect it will be much more common for
`Allocator` to override at least some of the other methods, like `fn
realloc`.)

The `alloc` method returns an `Address` when it succeeds, and
`dealloc` takes such an address as its input. But the client must also
provide metadata for the allocated block like its size and alignment.
This is encapsulated in the `Kind` argument to `alloc` and `dealloc`.

### Kinds of allocations

A `Kind` just carries the metadata necessary for satisfying an
allocation request. Its (current, private) representation is just a
size and alignment.

The more interesting thing about `Kind` is the
family of public methods associated with it for building new kinds via
composition; these are shown in the [kind api][].

### Reallocation Methods

Of course, real-world allocation often needs more than just
`alloc`/`dealloc`: in particular, one often wants to avoid extra
copying if the existing block of memory can be conceptually expanded
in place to meet new allocation needs. In other words, we want
`realloc`, plus alternatives to it that allow clients to avoid
round-tripping through the allocator API.

For this, the [memory reuse][] family of methods is appropriate.

### Type-based Helper Methods

Some readers might skim over the `Kind` API and immediately say "yuck,
all I wanted to do was allocate some nodes for a tree-structure and
let my clients choose how the backing memory is chosen! Why do I have
to wrestle with this `Kind` business?"

I agree with the sentiment; that's why the `Allocator` trait provides
a family of methods capturing [common usage patterns][].

## Unchecked variants

Finally, all of the methods above return `Result`, and guarantee some
amount of input validation. (This is largely because I observed code
duplication doing such validation on the client side; or worse, such
validation accidentally missing.)

However, some clients will want to bypass such checks (and do it
without risking undefined behavior by ensuring the preconditions hold
via local invariants in their container type).

For these clients, the `Allocator` trait provides
["unchecked" variants][unchecked variants] of nearly all of its
methods.

The idea here is that `Allocator` implementors are encouraged
to streamline the implmentations of such methods by assuming that all
of the preconditions hold.

 * However, to ease initial `impl Allocator` development for a given
   type, all of the unchecked methods have default implementations
   that call out to their checked counterparts.

 * (In other words, "unchecked" is in some sense a privilege being
   offered to impl's; but there is no guarantee that an arbitrary impl
   takes advantage of the privilege.)

## Why this API
[Why this API]: #why-this-api

Here are some quick points about how this API was selected

### Why not just `free(ptr)` for deallocation?

As noted in [RFC PR 39][] (and reiterated in [RFC PR 244][]), the basic `malloc` interface
{`malloc(size) -> ptr`, `free(ptr)`, `realloc(ptr, size) -> ptr`} is
lacking in a number of ways: `malloc` lacks the ability to request a
particular alignment, and `realloc` lacks the ability to express a
copy-free "reuse the input, or do nothing at all" request.  Another
problem with the `malloc` interface is that it burdens the allocator
with tracking the sizes of allocated data and re-extracting the
allocated size from the `ptr` in `free` and `realloc` calls (the
latter can be very cheap, but there is still no reason to pay that
cost in a language like Rust where the relevant size is often already
immediately available as a compile-time constant).

Therefore, in the name of (potential best-case) speed, we want to
require client code to provide the metadata like size and alignment
to both the allocation and deallocation call sites.

### Why not just `alloc`/`dealloc` (or `alloc`/`dealloc`/`realloc`)?

* The `alloc_one`/`dealloc_one` and `alloc_array`/`dealloc_array`
  capture a very common pattern for allocation of memory blocks where
  a simple value or array type is being allocated.

* The `alloc_array_unchecked` and `dealloc_array_unchecked` likewise
  capture a similar pattern, but are "less safe" in that they put more
  of an onus on the caller to validate the input parameters before
  calling the methods.

* The `alloc_excess` and `realloc_excess` methods provide a way for
  callers who can make use of excess memory to avoid unnecessary calls
  to `realloc`.

### Why the `Kind` abstraction?

While we do want to require clients to hand the allocator the size and
alignment, we have found that the code to compute such things follows
regular patterns. It makes more sense to factor those patterns out
into a common abstraction; this is what `Kind` provides: a high-level
API for describing the memory layout of a composite structure by
composing the layout of its subparts.

### Why return `Result` rather than a raw pointer?

My hypothesis is that the standard allocator API should embrace
`Result` as the standard way for describing local error conditions in
Rust.

In principle, we can use `Result` without adding *any* additional
overhead (at least in terms of the size of the values being returned
from the allocation calls), because the error type for the `Result`
can be zero-sized if so desired. That is why the error is an
associated type of the `Allocator`: allocators that want to ensure the
results have minimum size can use the zero-sized `RequestUnsatisfied`
or `MemoryExhausted` types as their associated `Self::Error`.

 * `RequestUnsatisfied` is a catch-all type that any allocator
   could use as its error type; doing so provides no hint to the
   client as to what they could do to try to service future memory
   requests.

 * `MemoryExhausted` is a specific error type meant for allocators
   that could in principle handle *any* sane input request, if there
   were sufficient memory available. (By "sane" we mean for example
   that the input arguments do not cause an arithmetic overflow during
   computation of the size of the memory block -- if they do, then it
   is reasonable for an allocator with this error type to respond that
   insufficent memory was available, rather than e.g. panicking.)

### Why return `Result` rather than directly `oom` on failure

Again, my hypothesis is that the standard allocator API should embrace
`Result` as the standard way for describing local error conditions in
Rust.

I want to leave it up to the clients to decide if they can respond to
out-of-memory (OOM) conditions on allocation failure.

However, since I also suspect that some programs would benefit from
contextual information about *which* allocator is reporting memory
exhaustion, I have made `oom` a method of the `Allocator` trait, so
that allocator clients can just call that on error (assuming they want
to trust the failure behavior of the allocator).

### Why is `usable_size` ever needed? Why not call `kind.size()` directly, as is done in the default implementation?

`kind.size()` returns the minimum required size that the client needs.
In a block-based allocator, this may be less than the *actual* size
that the allocator would ever provide to satisfy that `kind` of
request. Therefore, `usable_size` provides a way for clients to
observe what the minimum actual size of an allocated block for
that`kind` would be, for a given allocator.

(Note that the documentation does say that in general it is better for
clients to use `alloc_excess` and `realloc_excess` instead, if they
can, as a way to directly observe the *actual* amount of slop provided
by the particular allocator.)

### Why is `Allocator` an `unsafe trait`?

It just seems like a good idea given how much of the standard library
is going to assume that allocators are implemented according to their
specification.

(I had thought that `unsafe fn` for the methods would suffice, but
that is putting the burden of proof (of soundness) in the *wrong*
direction...)

## The GC integration strategy
[gc integration]: #the-gc-integration-strategy

One of the main reasons that [RFC PR 39] was not merged as written
was because it did not account for garbage collection (GC).

In particular, assuming that we eventually add support for GC in some
form, then any value that holds a reference to an object on the GC'ed
heap will need some linkage to the GC. In particular, if the *only*
such reference (i.e. the one with sole ownership) is held in a block
managed by a user-defined allocator, then we need to ensure that all
such references are found when the GC does its work.

The Rust project has control over the `libstd` provided allocators, so
the team can adapt them as necessary to fit the needs of whatever GC
designs come around. But the same is not true for user-defined
allocators: we want to ensure that adding support for them does not
inadvertantly kill any chance for adding GC later.

### The inspiration for Kind

Some aspects of the design of this RFC were selected in the hopes that
it would make such integration easier. In particular, the introduction
of the relatively high-level `Kind` abstraction was developed, in
part, as a way that a GC-aware allocator would build up a tracing
method associated with a kind.

Then I realized that the `Kind` abstraction may be valuable on its
own, without GC: It encapsulates important patterns when working with
representing data as memory records.

So, this RFC offers the `Kind` abstraction without promising that it
solves the GC problem. (It might, or it might not; we don't know yet.)

### Forwards-compatibility

So what *is* the solution for forwards-compatibility?

It is this: Rather than trying to build GC support into the
`Allocator` trait itself, we instead assume that when GC support
comes, it may come with a new trait (call it `GcAwareAllocator`).

 * (Perhaps we will instead use an attribute; the point is, whatever
   option we choose can be incorporated into the meta-data for a
   crate.)

Allocators that are are GC-compatible have to explicitly declare
themselves as such, by implementing `GcAwareAllocator`, which will
then impose new conditions on the methods of `Allocator`, for example
ensuring e.g. that allocated blocks of memory can be scanned
(i.e. "parsed") by the GC (if that in fact ends up being necessary).

This way, we can deploy an `Allocator` trait API today that does not
provide the necessary reflective hooks that a GC wuold need to access.

Crates that define their own `Allocator` implementations without also
claiming them to be GC-compatible will be forbidden from linking with
crates that require GC support. (In other words, when GC support
comes, we assume that the linking component of the Rust compiler will
be extended to check such compatibility requirements.)

# Drawbacks
[drawbacks]: #drawbacks

The API may be over-engineered.

The core set of methods (the ones without `unchecked`) return
`Result` and potentially impose unwanted input validation overhead.

 * The `_unchecked` variants are intended as the response to that,
   for clients who take care to validate the many preconditions
   themselves in order to minimize the allocation code paths.

# Alternatives
[alternatives]: #alternatives

## Just adopt [RFC PR 39][] with this RFC's GC strategy

The GC-compatibility strategy described here (in [gc integration][])
might work with a large number of alternative designs, such as that
from [RFC PR 39][].

While that is true, it seems like it would be a little short-sighted.
In particular, I have neither proven *nor* disproven the value of
`Kind` system described here with respect to GC integration.

As far as I know, it is the closest thing we have to a workable system
for allowing client code of allocators to accurately describe the
layout of values they are planning to allocate, which is the main
ingredient I believe to be necessary for the kind of dynamic
reflection that a GC will require of a user-defined allocator.

## Make `Kind` an associated type of `Allocator` trait

I explored making an `AllocKind` bound and then having

```rust
pub unsafe trait Allocator {
    /// Describes the sort of records that this allocator can
    /// construct.
    type Kind: AllocKind;

    ...
}
```

Such a design might indeed be workable. (I found it awkward, which is
why I abandoned it.)

But the question is: What benefit does it bring?

The main one I could imagine is that it might allow us to introduce a
division, at the type-system level, between two kinds of allocators:
those that are integrated with the GC (i.e., have an associated
`Allocator::Kind` that ensures that all allocated blocks are scannable
by a GC) and allocators that are *not* integrated with the GC (i.e.,
have an associated `Allocator::Kind` that makes no guarantees about
one will know how to scan the allocated blocks.

However, no such design has proven itself to be "obviously feasible to
implement," and therefore it would be unreasonable to make the `Kind`
an associated type of the `Allocator` trait without having at least a
few motivating examples that *are* clearly feasible and useful.

## Variations on the `Kind` API

 * Should `Kind` offer a `fn resize(&self, new_size: usize) -> Kind` constructor method?
   (Such a method would rule out deriving GC tracers from kinds; but we could
    maybe provide it as an `unsafe` method.)

 * Should `Kind` ensure an invariant that its associated size is
   always a multiple of its alignment?

   * Doing this would allow simplifying a small part of the API,
     namely the distinct `Kind::repeat` (returns both a kind and an
     offset) versus `Kind::array` (where the offset is derivable from
     the input `T`).

   * Such a constraint would have precendent; in particular, the
     `aligned_alloc` function of C11 requires the given size
     be a multiple of the alignment.

   * On the other hand, both the system and jemalloc allocators seem
     to support more flexible allocation patterns. Imposing the above
     invariant implies a certain loss of expressiveness over what we
     already provide today.

 * Should `Kind` ensure an invariant that its associated size is always positive?

   * Pro: Removes something that allocators would need to check about
     input kinds (the backing memory allocators will tend to require
     that the input sizes are positive).

   * Con: Requiring positive size means that zero-sized types do not have an associated
     `Kind`. That's not the end of the world, but it does make the `Kind` API slightly
     less convenient (e.g. one cannot use `extend` with a zero-sized kind to
     forcibly inject padding, because zero-sized kinds do not exist).

 * Should `Kind::align_to` add padding to the associated size? (Probably not; this would
   make it impossible to express certain kinds of patteerns.)

 * Should the `Kind` methods that might "fail" return `Result` instead of `Option`?

## Variations on the `Allocator` API

 * Should `Allocator::alloc` be safe instead of `unsafe fn`?
 
   * Clearly `fn dealloc` and `fn realloc` need to be `unsafe`, since
     feeding in improper inputs could cause unsound behavior. But is
     there any analogous input to `fn alloc` that could cause
     unsoundness (assuming that the `Kind` struct enforces invariants
     like "the associated size is non-zero")?

   * (I left it as `unsafe fn alloc` just to keep the API uniform with
     `dealloc` and `realloc`.)

 * Should `Allocator::realloc` not require that `new_kind.align()`
   evenly divide `kind.align()`? In particular, it is not too
   expensive to check if the two kinds are not compatible, and fall
   back on `alloc`/`dealloc` in that case.

 * Should `Allocator` not provide unchecked variants on `fn alloc`,
   `fn realloc`, et cetera? (To me it seems having them does no harm,
   apart from potentially misleading clients who do not read the
   documentation about what scenarios yield undefined behavior.

   * Another option here would be to provide a `trait
     UncheckedAllocator: Allocator` that carries the unchecked
     methods, so that clients who require such micro-optimized paths
     can ensure that their clients actually pass them an
     implementation that has the checks omitted.

 * On the flip-side of the previous bullet, should `Allocator` provide
   `fn alloc_one_unchecked` and `fn dealloc_one_unchecked` ?
   I think the only check that such variants would elide would be that
   `T` is not zero-sized; I'm not sure that's worth it.
   (But the resulting uniformity of the whole API might shift the
   balance to "worth it".)

# Unresolved questions
[unresolved]: #unresolved-questions

 * Should `Kind` be an associated type of `Allocator` (see
   [alternatives][] section for discussion).
   (In fact, most of the "Variations correspond to potentially
   unresolved questions.)

 * Should `dealloc` return a `Result` or not? (Under what
   circumstances would we expect `dealloc` to fail in a manner worth
   signalling? The main one I can think of is a transient failure,
   which is why the documentation for that method spends so much time
   discussing it.)

 * Are the type definitions for `Size`, `Capacity`, `Alignment`, and
   `Address` an abuse of the `NonZero` type? (Or do we just need some
   constructor for `NonZero` that asserts that the input is non-zero)?

 * Should `fn oom(&self)` take in more arguments (e.g. to allow the
   client to provide more contextual information about the OOM
   condition)?

 * Does `AllocError::is_transient` belong in this version of the API,
   or should we wait to add it later? (I originally suspected that
   libstd data types would want to make use of it, which would means
   we should add it.  However, in the absence of a concrete example
   stdlib type that would use it, we may be better off removing `fn
   is_transient` from this API (instead specifying that allocators
   with such transient failures will block (i.e. loop and retry
   internally), with the expectation that if a need for such an
   allocator does arise, we will then represent the API extension
   via a different trait (perhaps an extension trait of `Allocator`).

 * On that note, if we remove the `fn is_transient` method, should
   we get rid of the `AllocError` bound entirely? Is the given set
   of methods actually worth providing to all generic clients?

   (Keeping it seems very low cost to me; implementors can always opt
   to use the `MemoryExhausted` error type, which is cheap. But my
   intuition may be wrong.)
   
 * Do we need `Allocator::max_size` and `Allocator::max_align` ?
 
 * Should default impl of `Allocator::max_align` return `None`, or is
   there more suitable default? (perhaps e.g. `PLATFORM_PAGE_SIZE`?)

   The previous allocator documentation provided by Daniel Micay
   suggest that we should specify that behavior unspecified if
   allocation is too large, but if that is the case, then we should
   definitely provide some way to *observe* that threshold.)

   From what I can tell, we cannot currently assume that all
   low-level allocators will behave well for large alignments.
   See https://github.com/rust-lang/rust/issues/30170


# Appendices

## Bibliography
[Bibliography]: #bibliography

### RFC Pull Request #39: Allocator trait
[RFC PR 39]: https://github.com/rust-lang/rfcs/pull/39/files

Daniel Micay, 2014. RFC: Allocator trait. https://github.com/thestinger/rfcs/blob/ad4cdc2662cc3d29c3ee40ae5abbef599c336c66/active/0000-allocator-trait.md

### RFC Pull Request #244: Allocator RFC, take II
[RFC PR 244]: https://github.com/rust-lang/rfcs/pull/244

Felix Klock, 2014, Allocator RFC, take II, https://github.com/pnkfelix/rfcs/blob/d3c6068e823f495ee241caa05d4782b16e5ef5d8/active/0000-allocator.md

### Dynamic Storage Allocation: A Survey and Critical Review
Paul R. Wilson, Mark S. Johnstone, Michael Neely, and David Boles, 1995. [Dynamic Storage Allocation: A Survey and Critical Review](https://parasol.tamu.edu/~rwerger/Courses/689/spring2002/day-3-ParMemAlloc/papers/wilson95dynamic.pdf) ftp://ftp.cs.utexas.edu/pub/garbage/allocsrv.ps .  Slightly modified version appears in Proceedings of 1995 International Workshop on Memory Management (IWMM '95), Kinross, Scotland, UK, September 27--29, 1995 Springer Verlag LNCS

### Reconsidering custom memory allocation
[ReCustomMalloc]: http://dl.acm.org/citation.cfm?id=582421

Emery D. Berger, Benjamin G. Zorn, and Kathryn S. McKinley. 2002. [Reconsidering custom memory allocation][ReCustomMalloc]. In Proceedings of the 17th ACM SIGPLAN conference on Object-oriented programming, systems, languages, and applications (OOPSLA '02).

### The memory fragmentation problem: solved?
[MemFragSolvedP]: http://dl.acm.org/citation.cfm?id=286864

Mark S. Johnstone and Paul R. Wilson. 1998. [The memory fragmentation problem: solved?][MemFragSolvedP]. In Proceedings of the 1st international symposium on Memory management (ISMM '98).

### EASTL: Electronic Arts Standard Template Library
[EASTL]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2271.html

Paul Pedriana. 2007. [EASTL] -- Electronic Arts Standard Template Library. Document number: N2271=07-0131

### Towards a Better Allocator Model
[Halpern proposal]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1850.pdf

Pablo Halpern. 2005. [Towards a Better Allocator Model][Halpern proposal]. Document number: N1850=05-0110

### Various allocators

[jemalloc], [tcmalloc], [Hoard]

[jemalloc]: http://www.canonware.com/jemalloc/

[tcmalloc]: http://goog-perftools.sourceforge.net/doc/tcmalloc.html

[Hoard]: http://www.hoard.org/

[tracing garbage collector]: http://en.wikipedia.org/wiki/Tracing_garbage_collection

[malloc/free]: http://en.wikipedia.org/wiki/C_dynamic_memory_allocation

## ASCII art version of Allocator message sequence chart
[ascii-art]: #ascii-art-version-of-allocator-message-sequence-chart

This is an ASCII art version of the SVG message sequence chart
from the [semantics of allocators] section.

```
Program             Vec<Widget, &mut Allocator>         Allocator
  ||
  ||
   +--------------- create allocator -------------------> ** (an allocator is born)
  *| <------------ return allocator A ---------------------+
  ||                                                       |
  ||                                                       |
   +- create vec w/ &mut A -> ** (a vec is born)           |
  *| <------return vec V ------+                           |
  ||                           |                           |
   *------- push W_1 -------> *|                           |
   |                          ||                           |
   |                          ||                           |
   |                           +--- allocate W array ---> *|
   |                           |                          ||
   |                           |                          ||
   |                           |                           +---- (request system memory if necessary)
   |                           |                          *| <-- ...
   |                           |                          ||
   |                          *| <--- return *W block -----+
   |                          ||                           |
   |                          ||                           |
  *| <------- (return) -------+|                           |
  ||                           |                           |
   +------- push W_2 -------->+|                           |
   |                          ||                           |
  *| <------- (return) -------+|                           |
  ||                           |                           |
   +------- push W_3 -------->+|                           |
   |                          ||                           |
  *| <------- (return) -------+|                           |
  ||                           |                           |
   +------- push W_4 -------->+|                           |
   |                          ||                           |
  *| <------- (return) -------+|                           |
  ||                           |                           |
   +------- push W_5 -------->+|                           |
   |                          ||                           |
   |                           +---- realloc W array ---> *|
   |                           |                          ||
   |                           |                          ||
   |                           |                           +---- (request system memory if necessary)
   |                           |                          *| <-- ...
   |                           |                          ||
   |                          *| <--- return *W block -----+
  *| <------- (return) -------+|                           |
  ||                           |                           |
  ||                           |                           |
   .                           .                           .
   .                           .                           .
   .                           .                           .
  ||                           |                           |
  ||                           |                           |
  || (end of Vec scope)        |                           |
  ||                           |                           |
   +------ drop Vec --------> *|                           |
   |                          || (Vec destructor)          |
   |                          ||                           |
   |                           +---- dealloc W array -->  *|
   |                           |                          ||
   |                           |                           +---- (potentially return system memory)
   |                           |                          *| <-- ...
   |                           |                          ||
   |                          *| <------- (return) --------+
  *| <------- (return) --------+                           |
  ||                                                       |
  ||                                                       |
  ||                                                       |
  || (end of Allocator scope)                              |
  ||                                                       |
   +------------------ drop Allocator ------------------> *|
   |                                                      ||
   |                                                      |+---- (return any remaining associated memory)
   |                                                      *| <-- ...
   |                                                      ||
  *| <------------------ (return) -------------------------+
  ||
  ||
   .
   .
   .
```


## Transcribed Source for Allocator trait API
[Source for Allocator]: #transcribed-source-for-allocator-trait-api

Here is the whole source file for my prototype allocator API,
sub-divided roughly accordingly to functionality.

(We start with the usual boilerplate...)

```rust
// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "allocator_api",
            reason = "the precise API and guarantees it provides may be tweaked \
                      slightly, especially to possibly take into account the \
                      types being stored to make room for a future \
                      tracing garbage collector",
            issue = "27700")]

use core::cmp;
use core::fmt;
use core::mem;
use core::nonzero::NonZero;
use core::ptr::{self, Unique};

```

### Type Aliases
[type aliases]: #type-aliases

```rust
pub type Size = NonZero<usize>;
pub type Capacity = NonZero<usize>;
pub type Alignment = NonZero<usize>;

pub type Address = NonZero<*mut u8>;

/// Represents the combination of a starting address and
/// a total capacity of the returned block.
pub struct Excess(Address, Capacity);

fn size_align<T>() -> (usize, usize) {
    (mem::size_of::<T>(), mem::align_of::<T>())
}

```

### Kind API
[kind api]: #kind-api

```rust
/// Category for a memory record.
///
/// An instance of `Kind` describes a particular layout of memory.
/// You build a `Kind` up as an input to give to an allocator.
///
/// All kinds have an associated positive size; note that this implies
/// zero-sized types have no corresponding kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Kind {
    // size of the requested block of memory, measured in bytes.
    size: Size,
    // alignment of the requested block of memory, measured in bytes.
    // we ensure that this is always a power-of-two, because API's
    ///like `posix_memalign` require it and it is a reasonable
    // constraint to impose on Kind constructors.
    //
    // (However, we do not analogously require `align >= sizeof(void*)`,
    //  even though that is *also* a requirement of `posix_memalign`.)
    align: Alignment,
}


// FIXME: audit default implementations for overflow errors,
// (potentially switching to overflowing_add and
//  overflowing_mul as necessary).

impl Kind {
    // (private constructor)
    fn from_size_align(size: usize, align: usize) -> Kind {
        assert!(align.is_power_of_two()); 
        let size = unsafe { assert!(size > 0); NonZero::new(size) };
        let align = unsafe { assert!(align > 0); NonZero::new(align) };
        Kind { size: size, align: align }
    }

    /// The minimum size in bytes for a memory block of this kind.
    pub fn size(&self) -> NonZero<usize> { self.size }

    /// The minimum byte alignment for a memory block of this kind.
    pub fn align(&self) -> NonZero<usize> { self.align }

    /// Constructs a `Kind` suitable for holding a value of type `T`.
    /// Returns `None` if no such kind exists (e.g. for zero-sized `T`).
    pub fn new<T>() -> Option<Self> {
        let (size, align) = size_align::<T>();
        if size > 0 { Some(Kind::from_size_align(size, align)) } else { None }
    }

    /// Produces kind describing a record that could be used to
    /// allocate backing structure for `T` (which could be a trait
    /// or other unsized type like a slice).
    ///
    /// Returns `None` when no such kind exists; for example, when `x`
    /// is a reference to a zero-sized type.
    pub fn for_value<T: ?Sized>(t: &T) -> Option<Self> {
        let (size, align) = (mem::size_of_val(t), mem::align_of_val(t));
        if size > 0 {
            Some(Kind::from_size_align(size, align))
        } else {
            None
        }
    }

    /// Creates a kind describing the record that can hold a value
    /// of the same kind as `self`, but that also is aligned to
    /// alignment `align` (measured in bytes).
    ///
    /// If `self` already meets the prescribed alignment, then returns
    /// `self`.
    ///
    /// Note that this method does not add any padding to the overall
    /// size, regardless of whether the returned kind has a different
    /// alignment. In other words, if `K` has size 16, `K.align_to(32)`
    /// will *still* have size 16.
    pub fn align_to(&self, align: Alignment) -> Self {
        if align > self.align {
            let pow2_align = align.checked_next_power_of_two().unwrap();
            debug_assert!(pow2_align > 0); // (this follows from self.align > 0...)
            Kind { align: unsafe { NonZero::new(pow2_align) },
                   ..*self }
        } else {
            *self
        }
    }

    /// Returns the amount of padding we must insert after `self`
    /// to ensure that the following address will satisfy `align`
    /// (measured in bytes).
    ///
    /// Behavior undefined if `align` is not a power-of-two.
    ///
    /// Note that in practice, this is only useable if `align <=
    /// self.align` otherwise, the amount of inserted padding would
    /// need to depend on the particular starting address for the
    /// whole record, because `self.align` would not provide
    /// sufficient constraint.
    pub fn padding_needed_for(&self, align: Alignment) -> usize {
        debug_assert!(*align <= *self.align());
        let len = *self.size();
        let len_rounded_up = (len + *align - 1) & !(*align - 1);
        return len_rounded_up - len;
    }

    /// Creates a kind describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each to
    /// ensure that each instance is given its requested size and
    /// alignment. On success, returns `(k, offs)` where `k` is the
    /// kind of the array and `offs` is the distance between the start
    /// of each element in the array.
    ///
    /// On zero `n` or arithmetic overflow, returns `None`.
    pub fn repeat(&self, n: usize) -> Option<(Self, usize)> {
        if n == 0 { return None; }
        let padded_size = match self.size.checked_add(self.padding_needed_for(self.align)) {
            None => return None,
            Some(padded_size) => padded_size,
        };
        let alloc_size = match padded_size.checked_mul(n) {
            None => return None,
            Some(alloc_size) => alloc_size,
        };
        Some((Kind::from_size_align(alloc_size, *self.align), padded_size))
    }

    /// Creates a kind describing the record for `self` followed by
    /// `next`, including any necessary padding to ensure that `next`
    /// will be properly aligned. Note that the result kind will
    /// satisfy the alignment properties of both `self` and `next`.
    ///
    /// Returns `Some((k, offset))`, where `k` is kind of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// On arithmetic overflow, returns `None`.
    pub fn extend(&self, next: Self) -> Option<(Self, usize)> {
        let new_align = unsafe { NonZero::new(cmp::max(*self.align, *next.align)) };
        let realigned = Kind { align: new_align, ..*self };
        let pad = realigned.padding_needed_for(new_align);
        let offset = *self.size() + pad;
        let new_size = offset + *next.size();
        Some((Kind::from_size_align(new_size, *new_align), offset))
    }

    /// Creates a kind describing the record for `n` instances of
    /// `self`, with no padding between each instance.
    ///
    /// On zero `n` or overflow, returns `None`.
    pub fn repeat_packed(&self, n: usize) -> Option<Self> {
        let scaled = match self.size().checked_mul(n) {
            None => return None,
            Some(scaled) => scaled,
        };
        let size = unsafe { assert!(scaled > 0); NonZero::new(scaled) };
        Some(Kind { size: size, align: self.align })
    }

    /// Creates a kind describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incoporated *at all* into the resulting kind.
    ///
    /// Returns `(k, offset)`, where `k` is kind of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// (The `offset` is always the same as `self.size()`; we use this
    ///  signature out of convenience in matching the signature of
    ///  `fn extend`.)
    ///
    /// On arithmetic overflow, returns `None`.
    pub fn extend_packed(&self, next: Self) -> Option<(Self, usize)> {
        let new_size = match self.size().checked_add(*next.size()) {
            None => return None,
            Some(new_size) => new_size,
        };
        let new_size = unsafe { NonZero::new(new_size) };
        Some((Kind { size: new_size, ..*self }, *self.size()))
    }

    // Below family of methods *assume* inputs are pre- or
    // post-validated in some manner. (The implementations here
    ///do indirectly validate, but that is not part of their
    /// specification.)
    //
    // Since invalid inputs could yield ill-formed kinds, these
    // methods are `unsafe`.

    /// Creates kind describing the record for a single instance of `T`.
    /// Requires `T` has non-zero size.
    pub unsafe fn new_unchecked<T>() -> Self {
        let (size, align) = size_align::<T>();
        Kind::from_size_align(size, align)
    }


    /// Creates a kind describing the record for `self` followed by
    /// `next`, including any necessary padding to ensure that `next`
    /// will be properly aligned. Note that the result kind will
    /// satisfy the alignment properties of both `self` and `next`.
    ///
    /// Returns `(k, offset)`, where `k` is kind of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// Requires no arithmetic overflow from inputs.
    pub unsafe fn extend_unchecked(&self, next: Self) -> (Self, usize) {
        self.extend(next).unwrap()
    }

    /// Creates a kind describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each.
    ///
    /// Requires non-zero `n` and no arithmetic overflow from inputs.
    /// (See also the `fn array` checked variant.)
    pub unsafe fn repeat_unchecked(&self, n: usize) -> (Self, usize) {
        self.repeat(n).unwrap()
    }

    /// Creates a kind describing the record for `n` instances of
    /// `self`, with no padding between each instance.
    ///
    /// Requires non-zero `n` and no arithmetic overflow from inputs.
    /// (See also the `fn array_packed` checked variant.)
    pub unsafe fn repeat_packed_unchecked(&self, n: usize) -> Self {
        self.repeat_packed(n).unwrap()
    }

    /// Creates a kind describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incoporated *at all* into the resulting kind.
    ///
    /// Returns `(k, offset)`, where `k` is kind of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// (The `offset` is always the same as `self.size()`; we use this
    ///  signature out of convenience in matching the signature of
    ///  `fn extend`.)
    ///
    /// Requires no arithmetic overflow from inputs.
    /// (See also the `fn extend_packed` checked variant.)
    pub unsafe fn extend_packed_unchecked(&self, next: Self) -> (Self, usize) {
        self.extend_packed(next).unwrap()
    }

    /// Creates a kind describing the record for a `[T; n]`.
    ///
    /// On zero `n`, zero-sized `T`, or arithmetic overflow, returns `None`.
    pub fn array<T>(n: usize) -> Option<Self> {
        Kind::new::<T>()
            .and_then(|k| k.repeat(n))
            .map(|(k, offs)| {
                debug_assert!(offs == mem::size_of::<T>());
                k
            })
    }

    /// Creates a kind describing the record for a `[T; n]`.
    ///
    /// Requires nonzero `n`, nonzero-sized `T`, and no arithmetic
    /// overflow; otherwise behavior undefined.
    pub fn array_unchecked<T>(n: usize) -> Self {
        Kind::array::<T>(n).unwrap()
    }

}

```

### AllocError API
[error api]: #allocerror-api

```rust
/// `AllocError` instances provide feedback about the cause of an allocation failure.
pub trait AllocError {
    /// Construct an error that indicates operation failure due to
    /// invalid input values for the request.
    ///
    /// This can be used, for example, to signal an overflow occurred
    /// during arithmetic computation. (However, since overflows
    /// frequently represent an allocation attempt that would exhaust
    /// memory, clients are alternatively allowed to constuct an error
    /// representing memory exhaustion in such scenarios.)
    fn invalid_input() -> Self;

    /// Returns true if the error is due to hitting some resource
    /// limit or otherwise running out of memory. This condition
    /// strongly implies that *some* series of deallocations would
    /// allow a subsequent reissuing of the original allocation
    /// request to succeed.
    ///
    /// Exhaustion is a common interpretation of an allocation failure;
    /// e.g. usually when `malloc` returns `null`, it is because of
    /// hitting a user resource limit or system memory exhaustion.
    ///
    /// Note that the resource exhaustion could be specific to the
    /// original allocator (i.e. the only way to free up memory is by
    /// deallocating memory attached to that allocator), or it could
    /// be associated with some other state outside of the original
    /// alloactor. The `AllocError` trait does not distinguish between
    /// the two scenarios.
    ///
    /// Finally, error responses to allocation input requests that are
    /// *always* illegal for *any* allocator (e.g. zero-sized or
    /// arithmetic-overflowing requests) are allowed to respond `true`
    /// here. (This is to allow `MemoryExhausted` as a valid error type
    /// for an allocator that can handle all "sane" requests.)
    fn is_memory_exhausted(&self) -> bool;

    /// Returns true if the allocator is fundamentally incapable of
    /// satisfying the original request. This condition implies that
    /// such an allocation request will never succeed on this
    /// allocator, regardless of environment, memory pressure, or
    /// other contextual condtions.
    ///
    /// An example where this might arise: A block allocator that only
    /// supports satisfying memory requests where each allocated block
    /// is at most `K` bytes in size.
    fn is_request_unsupported(&self) -> bool;

    /// Returns true only if the error is transient. "Transient" is
    /// meant here in the sense that there is a reasonable chance that
    /// re-issuing the same allocation request in the future *could*
    /// succeed, even if nothing else changes about the overall
    /// context of the request.
    ///
    /// An example where this might arise: An allocator shared across
    /// threads that fails upon detecting interference (rather than
    /// e.g. blocking).
    fn is_transient(&self) -> bool { false } // most errors are not transient
}

/// The `MemoryExhausted` error represents a blanket condition
/// that the given request was not satisifed for some reason beyond
/// any particular limitations of a given allocator.
///
/// It roughly corresponds to getting `null` back from a call to `malloc`:
/// you've probably exhausted memory (though there might be some other
/// explanation; see discussion with `AllocError::is_memory_exhausted`).
///
/// Allocators that can in principle allocate any kind of legal input
/// might choose this as their associated error type.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MemoryExhausted;

/// The `AllocErr` error specifies whether an allocation failure is
/// specifically due to resource exhaustion or if it is due to
/// something wrong when combining the given input arguments with this
/// allocator.

/// Allocators that only support certain classes of inputs might choose this
/// as their associated error type, so that clients can respond appropriately
/// to specific error failure scenarios.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AllocErr {
    /// Error due to hitting some resource limit or otherwise running
    /// out of memory. This condition strongly implies that *some*
    /// series of deallocations would allow a subsequent reissuing of
    /// the original allocation request to succeed.
    Exhausted,

    /// Error due to allocator being fundamentally incapable of
    /// satisfying the original request. This condition implies that
    /// such an allocation request will never succeed on the given
    /// allocator, regardless of environment, memory pressure, or
    /// other contextual condtions.
    Unsupported,
}

impl AllocError for MemoryExhausted {
    fn invalid_input() -> Self { MemoryExhausted }
    fn is_memory_exhausted(&self) -> bool { true }
    fn is_request_unsupported(&self) -> bool { false }
}

impl AllocError for AllocErr {
    fn invalid_input() -> Self { AllocErr::Unsupported }
    fn is_memory_exhausted(&self) -> bool { *self == AllocErr::Exhausted }
    fn is_request_unsupported(&self) -> bool { *self == AllocErr::Unsupported }
}

```

### Allocator trait header
[trait header]: #allocator-trait-header

```rust
/// An implementation of `Allocator` can allocate, reallocate, and
/// deallocate arbitrary blocks of data described via `Kind`.
///
/// Some of the methods require that a kind *fit* a memory block.
/// What it means for a kind to "fit" a memory block means is that
/// the following two conditions must hold:
///
/// 1. The block's starting address must be aligned to `kind.align()`.
///
/// 2. The block's size must fall in the range `[orig, usable]`, where:
///
///    * `orig` is the size last used to allocate the block, and
///
///    * `usable` is the capacity that was (or would have been)
///      returned when (if) the block was allocated via a call to
///      `alloc_excess` or `realloc_excess`.
///
/// Note that due to the constraints in the methods below, a
/// lower-bound on `usable` can be safely approximated by a call to
/// `usable_size`.
pub unsafe trait Allocator {
    /// When allocation requests cannot be satisified, an instance of
    /// this error is returned.
    ///
    /// Many allocators will want to use the zero-sized
    /// `MemoryExhausted` type for this.
    type Error: AllocError + fmt::Debug;

```

### Allocator core alloc and dealloc
[alloc and dealloc]: #allocator-core-alloc-and-dealloc

```rust
    /// Returns a pointer suitable for holding data described by
    /// `kind`, meeting its size and alignment guarantees.
    ///
    /// The returned block of storage may or may not have its contents
    /// initialized. (Extension subtraits might restrict this
    /// behavior, e.g. to ensure initialization.)
    ///
    /// Returns `Err` if allocation fails or if `kind` does
    /// not meet allocator's size or alignment constraints.
    unsafe fn alloc(&mut self, kind: Kind) -> Result<Address, Self::Error>;

    /// Deallocate the memory referenced by `ptr`.
    ///
    /// `ptr` must have previously been provided via this allocator,
    /// and `kind` must *fit* the provided block (see above);
    /// otherwise yields undefined behavior.
    ///
    /// Returns `Err` only if deallocation fails in some fashion. If
    /// the returned error is *transient*, then ownership of the
    /// memory block is transferred back to the caller (see
    /// `AllocError::is_transient`). Otherwise, callers must assume
    /// that ownership of the block has been unrecoverably lost.
    ///
    /// Note: Implementors are encouraged to avoid `Err`-failure from
    /// `dealloc`; most memory allocation APIs do not support
    /// signalling failure in their `free` routines, and clients are
    /// likely to incorporate that assumption into their own code and
    /// just `unwrap` the result of this call.
    unsafe fn dealloc(&mut self, ptr: Address, kind: Kind) -> Result<(), Self::Error>;

    /// Allocator-specific method for signalling an out-of-memory
    /// condition.
    ///
    /// Any activity done by the `oom` method should ensure that it
    /// does not infinitely regress in nested calls to `oom`. In
    /// practice this means implementors should eschew allocating,
    /// especially from `self` (directly or indirectly).
    ///
    /// Implementors of this trait are discouraged from panicking or
    /// aborting from other methods in the event of memory exhaustion;
    /// instead they should return an appropriate error from the
    /// invoked method, and let the client decide whether to invoke
    /// this `oom` method.
    unsafe fn oom(&mut self) -> ! { ::core::intrinsics::abort() }

```

### Allocator-specific quantities and limits
[quantites and limits]: #allocator-specific-quantities-and-limits

```rust
    // == ALLOCATOR-SPECIFIC QUANTITIES AND LIMITS ==
    // max_size, max_align, usable_size

    /// The maximum requestable size in bytes for memory blocks
    /// managed by this allocator.
    ///
    /// Returns `None` if this allocator has no explicit maximum size.
    /// (Note that such allocators may well still have an *implicit*
    /// maximum size; i.e. allocation requests can always fail.)
    fn max_size(&self) -> Option<Size> { None }

    /// The maximum requestable alignment in bytes for memory blocks
    /// managed by this allocator.
    ///
    /// Returns `None` if this allocator has no assigned maximum
    /// alignment.  (Note that such allocators may well still have an
    /// *implicit* maximum alignment; i.e. allocation requests can
    /// always fail.)
    fn max_align(&self) -> Option<Alignment> { None }

    /// Returns the minimum guaranteed usable size of a successful
    /// allocation created with the specified `kind`.
    ///
    /// Clients who wish to make use of excess capacity are encouraged
    /// to use the `alloc_excess` and `realloc_excess` instead, as
    /// this method is constrained to conservatively report a value
    /// less than or equal to the minimum capacity for *all possible*
    /// calls to those methods.
    ///
    /// However, for clients that do not wish to track the capacity
    /// returned by `alloc_excess` locally, this method is likely to
    /// produce useful results.
    unsafe fn usable_size(&self, kind: Kind) -> Capacity { kind.size() }

```

### Allocator methods for memory reuse
[memory reuse]: #allocator-methods-for-memory-reuse

```rust
    // == METHODS FOR MEMORY REUSE ==
    // realloc. alloc_excess, realloc_excess
    
    /// Returns a pointer suitable for holding data described by
    /// `new_kind`, meeting its size and alignment guarantees. To
    /// accomplish this, this may extend or shrink the allocation
    /// referenced by `ptr` to fit `new_kind`.
    ///
    /// * `ptr` must have previously been provided via this allocator.
    ///
    /// * `kind` must *fit* the `ptr` (see above). (The `new_kind`
    ///   argument need not fit it.)
    ///
    /// Behavior undefined if either of latter two constraints are unmet.
    ///
    /// In addition, `new_kind` should not impose a stronger alignment
    /// constraint than `kind`. (In other words, `new_kind.align()`
    /// must evenly divide `kind.align()`; note this implies the
    /// alignment of `new_kind` must not exceed that of `kind`.)
    /// However, behavior is well-defined (though underspecified) when
    /// this constraint is violated; further discussion below.
    ///
    /// If this returns `Ok`, then ownership of the memory block
    /// referenced by `ptr` has been transferred to this
    /// allocator. The memory may or may not have been freed, and
    /// should be considered unusable (unless of course it was
    /// transferred back to the caller again via the return value of
    /// this method).
    ///
    /// Returns `Err` only if `new_kind` does not meet the allocator's
    /// size and alignment constraints of the allocator or the
    /// alignment of `kind`, or if reallocation otherwise fails. (Note
    /// that did not say "if and only if" -- in particular, an
    /// implementation of this method *can* return `Ok` if
    /// `new_kind.align() > old_kind.align()`; or it can return `Err`
    /// in that scenario.)
    ///
    /// If this method returns `Err`, then ownership of the memory
    /// block has not been transferred to this allocator, and the
    /// contents of the memory block are unaltered.
    unsafe fn realloc(&mut self,
                      ptr: Address,
                      kind: Kind,
                      new_kind: Kind) -> Result<Address, Self::Error> {
        // All Kind alignments are powers of two, so a comparison
        // suffices here (rather than resorting to a `%` operation).
        if new_kind.size() <= self.usable_size(kind) && new_kind.align() <= kind.align() {
            return Ok(ptr);
        } else {
            let result = self.alloc(new_kind);
            if let Ok(new_ptr) = result {
                ptr::copy(*ptr as *const u8, *new_ptr, cmp::min(*kind.size(), *new_kind.size()));
                loop {
                    if let Err(err) = self.dealloc(ptr, kind) {
                        // all we can do from the realloc abstraction
                        // is either:
                        //
                        // 1. free the block we just finished copying
                        //    into and pass the error up,
                        // 2. ignore the dealloc error, or
                        // 3. try again.
                        //
                        // They are all terrible; 1 seems unjustifiable.
                        // So we choose 2, unless the error is transient.
                        if err.is_transient() { continue; }
                    }
                    break;
                }
            }
            result
        }
    }

    /// Behaves like `fn alloc`, but also returns the whole size of
    /// the returned block. For some `kind` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    unsafe fn alloc_excess(&mut self, kind: Kind) -> Result<Excess, Self::Error> {
        self.alloc(kind).map(|p| Excess(p, self.usable_size(kind)))
    }

    /// Behaves like `fn realloc`, but also returns the whole size of
    /// the returned block. For some `kind` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    unsafe fn realloc_excess(&mut self,
                             ptr: Address,
                             kind: Kind,
                             new_kind: Kind) -> Result<Excess, Self::Error> {
        self.realloc(ptr, kind, new_kind)
            .map(|p| Excess(p, self.usable_size(new_kind)))
    }

```

### Allocator common usage patterns
[common usage patterns]: #allocator-common-usage-patterns

```rust
    // == COMMON USAGE PATTERNS ==
    // alloc_one, dealloc_one, alloc_array, realloc_array. dealloc_array
    
    /// Allocates a block suitable for holding an instance of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `alloc`/`realloc` methods of this allocator.
    unsafe fn alloc_one<T>(&mut self) -> Result<Unique<T>, Self::Error> {
        if let Some(k) = Kind::new::<T>() {
            self.alloc(k).map(|p|Unique::new(*p as *mut T))
        } else {
            // (only occurs for zero-sized T)
            debug_assert!(mem::size_of::<T>() == 0);
            Err(Self::Error::invalid_input())
        }
    }

    /// Deallocates a block suitable for holding an instance of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    unsafe fn dealloc_one<T>(&mut self, mut ptr: Unique<T>) -> Result<(), Self::Error> {
        let raw_ptr = NonZero::new(ptr.get_mut() as *mut T as *mut u8);
        self.dealloc(raw_ptr, Kind::new::<T>().unwrap())
    }

    /// Allocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `alloc`/`realloc` methods of this allocator.
    unsafe fn alloc_array<T>(&mut self, n: usize) -> Result<Unique<T>, Self::Error> {
        match Kind::array::<T>(n) {
            Some(kind) => self.alloc(kind).map(|p|Unique::new(*p as *mut T)),
            None => Err(Self::Error::invalid_input()),
        }
    }

    /// Reallocates a block previously suitable for holding `n_old`
    /// instances of `T`, returning a block suitable for holding
    /// `n_new` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `alloc`/`realloc` methods of this allocator.
    unsafe fn realloc_array<T>(&mut self,
                               ptr: Unique<T>,
                               n_old: usize,
                               n_new: usize) -> Result<Unique<T>, Self::Error> {
        let old_new_ptr = (Kind::array::<T>(n_old), Kind::array::<T>(n_new), *ptr);
        if let (Some(k_old), Some(k_new), ptr) = old_new_ptr {
            self.realloc(NonZero::new(ptr as *mut u8), k_old, k_new)
                .map(|p|Unique::new(*p as *mut T))
        } else {
            Err(Self::Error::invalid_input())
        }
    }

    /// Deallocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    unsafe fn dealloc_array<T>(&mut self, ptr: Unique<T>, n: usize) -> Result<(), Self::Error> {
        let raw_ptr = NonZero::new(*ptr as *mut u8);
        if let Some(k) = Kind::array::<T>(n) {
            self.dealloc(raw_ptr, k)
        } else {
            Err(Self::Error::invalid_input())
        }
    }

```

### Allocator unchecked method variants
[unchecked variants]: #allocator-unchecked-method-variants

```rust
    // UNCHECKED METHOD VARIANTS

    /// Returns a pointer suitable for holding data described by
    /// `kind`, meeting its size and alignment guarantees.
    ///
    /// The returned block of storage may or may not have its contents
    /// initialized. (Extension subtraits might restrict this
    /// behavior, e.g. to ensure initialization.)
    ///
    /// Returns `None` if request unsatisfied.
    ///
    /// Behavior undefined if input does not meet size or alignment
    /// constraints of this allocator.
    unsafe fn alloc_unchecked(&mut self, kind: Kind) -> Option<Address> {
        // (default implementation carries checks, but impl's are free to omit them.)
        self.alloc(kind).ok()
    }

    /// Deallocate the memory referenced by `ptr`.
    ///
    /// `ptr` must have previously been provided via this allocator,
    /// and `kind` must *fit* the provided block (see above).
    /// Otherwise yields undefined behavior.
    unsafe fn dealloc_unchecked(&mut self, ptr: Address, kind: Kind) {
        // (default implementation carries checks, but impl's are free to omit them.)
        self.dealloc(ptr, kind).unwrap()
    }

    /// Returns a pointer suitable for holding data described by
    /// `new_kind`, meeting its size and alignment guarantees. To
    /// accomplish this, may extend or shrink the allocation
    /// referenced by `ptr` to fit `new_kind`.
    ////
    /// (In other words, ownership of the memory block associated with
    /// `ptr` is first transferred back to this allocator, but the
    /// same block may or may not be transferred back as the result of
    /// this call.)
    ///
    /// * `ptr` must have previously been provided via this allocator.
    ///
    /// * `kind` must *fit* the `ptr` (see above). (The `new_kind`
    ///   argument need not fit it.)
    ///
    /// * `new_kind` must meet the allocator's size and alignment
    ///    constraints. In addition, `new_kind.align()` must equal
    ///    `kind.align()`. (Note that this is a stronger constraint
    ///    that that imposed by `fn realloc`.)
    ///
    /// Behavior undefined if any of latter three constraints are unmet.
    ///
    /// If this returns `Some`, then the memory block referenced by
    /// `ptr` may have been freed and should be considered unusable.
    ///
    /// Returns `None` if reallocation fails; in this scenario, the
    /// original memory block referenced by `ptr` is unaltered.
    unsafe fn realloc_unchecked(&mut self,
                                ptr: Address,
                                kind: Kind,
                                new_kind: Kind) -> Option<Address> {
        // (default implementation carries checks, but impl's are free to omit them.)
        self.realloc(ptr, kind, new_kind).ok()
    }

    /// Behaves like `fn alloc_unchecked`, but also returns the whole
    /// size of the returned block. 
    unsafe fn alloc_excess_unchecked(&mut self, kind: Kind) -> Option<Excess> {
        self.alloc_excess(kind).ok()
    }

    /// Behaves like `fn realloc_unchecked`, but also returns the
    /// whole size of the returned block.
    unsafe fn realloc_excess_unchecked(&mut self,
                                       ptr: Address,
                                       kind: Kind,
                                       new_kind: Kind) -> Option<Excess> {
        self.realloc_excess(ptr, kind, new_kind).ok()
    }


    /// Allocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// Requires inputs are non-zero and do not cause arithmetic
    /// overflow, and `T` is not zero sized; otherwise yields
    /// undefined behavior.
    unsafe fn alloc_array_unchecked<T>(&mut self, n: usize) -> Option<Unique<T>> {
        let kind = Kind::array_unchecked::<T>(n);
        self.alloc_unchecked(kind).map(|p|Unique::new(*p as *mut T))
    }

    /// Reallocates a block suitable for holding `n_old` instances of `T`,
    /// returning a block suitable for holding `n_new` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// Requires inputs are non-zero and do not cause arithmetic
    /// overflow, and `T` is not zero sized; otherwise yields
    /// undefined behavior.
    unsafe fn realloc_array_unchecked<T>(&mut self,
                                         ptr: Unique<T>,
                                         n_old: usize,
                                         n_new: usize) -> Option<Unique<T>> {
        let (k_old, k_new, ptr) = (Kind::array_unchecked::<T>(n_old),
                                   Kind::array_unchecked::<T>(n_new),
                                   *ptr);
        self.realloc_unchecked(NonZero::new(ptr as *mut u8), k_old, k_new)
            .map(|p|Unique::new(*p as *mut T))
    }

    /// Deallocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// Requires inputs are non-zero and do not cause arithmetic
    /// overflow, and `T` is not zero sized; otherwise yields
    /// undefined behavior.
    unsafe fn dealloc_array_unchecked<T>(&mut self, ptr: Unique<T>, n: usize) {
        let kind = Kind::array_unchecked::<T>(n);
        self.dealloc_unchecked(NonZero::new(*ptr as *mut u8), kind);
    }
}
```
