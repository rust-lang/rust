- Feature Name: allocator_api
- Start Date: 2015-12-01
- RFC PR: https://github.com/rust-lang/rfcs/pull/1398
- Rust Issue: https://github.com/rust-lang/rust/issues/32838

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
the core type machinery and language idioms (e.g. using `Result` to
propagate dynamic error conditions), and provides
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

The source code for the `Allocator` trait prototype is provided in an
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

 * The metadata for any allocation is captured in a `Layout`
   abstraction. This type carries (at minimum) the size and alignment
   requirements for a memory request.

   * The `Layout` type provides a large family of functional construction
     methods for building up the description of how memory is laid out.

     * Any sized type `T` can be mapped to its `Layout`, via `Layout::new::<T>()`,

     * Heterogenous structure; e.g. `layout1.extend(layout2)`,

     * Homogenous array types: `layout.repeat(n)` (for `n: usize`),

     * There are packed and unpacked variants for the latter two methods.

   * Helper `Allocator` methods like `fn alloc_one` and `fn
     alloc_array` allow client code to interact with an allocator
     without ever directly constructing a `Layout`.

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

   FIXME: `RefCell<Pool>` is not going to work with the allocator API
   envisaged here; see [comment from gankro][]. We will need to
   address this (perhaps just by pointing out that it is illegal and
   suggesting a standard pattern to work around it) before this RFC
   can be accepted.

[comment from gankro]: https://github.com/rust-lang/rfcs/pull/1398#issuecomment-162681096

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
#[derive(Debug)]
pub struct DumbBumpPool {
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
    pub fn new(name: &'static str,
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

Here are some other design choices of note:

 * Our Bump Allocator is going to use a most simple-minded deallocation
   policy: calls to `fn dealloc` are no-ops. Instead, every request takes
   up fresh space in the backing storage, until the pool is exhausted.
   (This was one reason I use the word "Dumb" in its name.)

 * Since we want to be able to share the bump-allocator amongst multiple
   (lifetime-scoped) threads, we will implement the `Allocator` interface
   as a *handle* pointing to the pool; in this case, a simple reference.

 * Since the whole point of this particular bump-allocator is to
   shared across threads (otherwise there would be no need to use
   `AtomicPtr` for the `avail` field), we will want to implement the
   (unsafe) `Sync` trait on it (doing this signals that it is safe to
   send `&DumbBumpPool` to other threads).

Here is that `impl Sync`.

```rust
/// Note of course that this impl implies we must review all other
/// code for DumbBumpPool even more carefully.
unsafe impl Sync for DumbBumpPool { }
```

Here is the demo implementation of `Allocator` for the type.

```rust
unsafe impl<'a> Allocator for &'a DumbBumpPool {
    unsafe fn alloc(&mut self, layout: alloc::Layout) -> Result<Address, AllocErr> {
        let align = layout.align();
        let size = layout.size();

        let mut curr_addr = self.avail.load(Ordering::Relaxed);
        loop {
            let curr = curr_addr as usize;
            let (sum, oflo) = curr.overflowing_add(align - 1);
            let curr_aligned = sum & !(align - 1);
            let remaining = (self.end as usize) - curr_aligned;
            if oflo || remaining < size {
                return Err(AllocErr::Exhausted { request: layout.clone() });
            }

            let curr_aligned = curr_aligned as *mut u8;
            let new_curr = curr_aligned.offset(size as isize);

            let attempt = self.avail.compare_and_swap(curr_addr, new_curr, Ordering::Relaxed);
            // If the allocation attempt hits interference ...
            if curr_addr != attempt {
                curr_addr = attempt;
                continue; // .. then try again
            } else {
                println!("alloc finis ok: 0x{:x} size: {}", curr_aligned as usize, size);
                return Ok(curr_aligned);
            }
        }
    }

    unsafe fn dealloc(&mut self, _ptr: Address, _layout: alloc::Layout) {
        // this bump-allocator just no-op's on dealloc
    }

    fn oom(&mut self, err: AllocErr) -> ! {
        let remaining = self.end as usize - self.avail.load(Ordering::Relaxed) as usize;
        panic!("exhausted memory in {} on request {:?} with avail: {}; self: {:?}",
               self.name, err, remaining, self);
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

### What about standard library containers?

The intention of this RFC is that the Rust standard library will be
extended with parameteric allocator support: `Vec`, `HashMap`, etc
should all eventually be extended with the ability to use an
alternative allocator for their backing storage.

However, this RFC does not prescribe when or how this should happen.

Under the design of this RFC, Allocators parameters are specified via
a *generic type parameter* on the container type. This strongly
implies that `Vec<T>` and `HashMap<K, V>` will need to be extended
with an allocator type parameter, i.e.: `Vec<T, A:Allocator>` and
`HashMap<K, V, A:Allocator>`.

There are two reasons why such extension is left to later work, after
this RFC.

#### Default type parameter fallback

On its own, such a change would be backwards incompatible (i.e. a huge
breaking change), and also would simply be just plain inconvenient for
typical use cases. Therefore, the newly added type parameters will
almost certainly require a *default type*: `Vec<T:
A:Allocator=HeapAllocator>` and
`HashMap<K,V,A:Allocator=HeapAllocator>`.

Default type parameters themselves, in the context of type defintions,
are a stable part of the Rust language.

However, the exact semantics of how default type parameters interact
with inference is still being worked out (in part *because* allocators
are a motivating use case), as one can see by reading the following:

* RFC 213, "Finalize defaulted type parameters": https://github.com/rust-lang/rfcs/blob/master/text/0213-defaulted-type-params.md

 * Tracking Issue for RFC 213: Default Type Parameter Fallback: https://github.com/rust-lang/rust/issues/27336

* Feature gate defaulted type parameters appearing outside of types: https://github.com/rust-lang/rust/pull/30724

#### Fully general container integration needs Dropck Eyepatch

The previous problem was largely one of programmer
ergonomics. However, there is also a subtle soundness issue that
arises due to an current implementation artifact.

Standard library types like `Vec<T>` and `HashMap<K,V>` allow
instantiating the generic parameters `T`, `K`, `V` with types holding
lifetimes that do not strictly outlive that of the container itself.
(I will refer to such instantiations of `Vec` and `HashMap`
"same-lifetime instances" as a shorthand in this discussion.)

Same-lifetime instance support is currently implemented for `Vec` and
`HashMap` via an unstable attribute that is too
coarse-grained. Therefore, we cannot soundly add the allocator
parameter to `Vec` and `HashMap` while also continuing to allow
same-lifetime instances without first addressing this overly coarse
attribute. I have an open RFC to address this, the "Dropck Eyepatch"
RFC; that RFC explains in more detail why this problem arises, using
allocators as a specific motivating use case.

 * Concrete code illustrating this exact example (part of Dropck Eyepatch RFC):
   https://github.com/pnkfelix/rfcs/blob/dropck-eyepatch/text/0000-dropck-param-eyepatch.md#example-vect-aallocatordefaultallocator

 * Nonparametric dropck RFC https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md

#### Standard library containers conclusion

Rather than wait for the above issues to be resolved, this RFC
proposes that we at least stabilize the `Allocator` trait interface;
then we will at least have a starting point upon which to prototype
standard library integration.

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
the `Allocator` trait assumes/requires all of the following conditions.
(Note: this list of conditions uses the phrases "should", "must", and "must not"
in a formal manner, in the style of [IETF RFC 2119][].)

[IETF RFC 2119]: https://www.ietf.org/rfc/rfc2119.txta

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
     impl Allocator for MegaEmbedded { ... } // INVALID IMPL
     ```
     The latter impl is simply unreasonable (at least if one is
     intending to satisfy requests by returning pointers into
     `self.bytes`).

     Note that an allocator that owns its pool *indirectly*
     (i.e. does not have the pool's state embedded in the allocator) is fine:
     ```rust
     struct MegaIndirect { pool: *mut [u8; 1024*1024], cursor: usize, ... }
     impl Allocator for MegaIndirect { ... } // OKAY
     ```

     (I originally claimed that `impl Allocator for &mut MegaEmbedded`
     would also be a legal example of an allocator that is an indirect handle
     to an unembedded pool, but others pointed out that handing out the
     addresses pointing into that embedded pool could end up violating our
     aliasing rules for `&mut`. I obviously did not expect that outcome; I
     would be curious to see what the actual design space is here.)

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

### Basic implementation

An instance of an allocator has many methods, but an implementor of
the trait need only provide two method bodies: [alloc and dealloc][].

(This is only *somewhat* analogous to the `Iterator` trait in Rust. It
is currently very uncommon to override any methods of `Iterator` except
for `fn next`. However, I expect it will be much more common for
`Allocator` to override at least some of the other methods, like `fn
realloc`.)

The `alloc` method returns an `Address` when it succeeds, and
`dealloc` takes such an address as its input. But the client must also
provide metadata for the allocated block like its size and alignment.
This is encapsulated in the `Layout` argument to `alloc` and `dealloc`.

### Memory layouts

A `Layout` just carries the metadata necessary for satisfying an
allocation request. Its (current, private) representation is just a
size and alignment.

The more interesting thing about `Layout` is the
family of public methods associated with it for building new layouts via
composition; these are shown in the [layout api][].

### Reallocation Methods

Of course, real-world allocation often needs more than just
`alloc`/`dealloc`: in particular, one often wants to avoid extra
copying if the existing block of memory can be conceptually expanded
in place to meet new allocation needs. In other words, we want
`realloc`, plus alternatives to it (`alloc_excess`) that allow clients to avoid
round-tripping through the allocator API.

For this, the [memory reuse][] family of methods is appropriate.

### Type-based Helper Methods

Some readers might skim over the `Layout` API and immediately say "yuck,
all I wanted to do was allocate some nodes for a tree-structure and
let my clients choose how the backing memory is chosen! Why do I have
to wrestle with this `Layout` business?"

I agree with the sentiment; that's why the `Allocator` trait provides
a family of methods capturing [common usage patterns][],
for example, `a.alloc_one::<T>()` will return a `Unique<T>` (or error).

## Unchecked variants

Almost all of the methods above return `Result`, and guarantee some
amount of input validation. (This is largely because I observed code
duplication doing such validation on the client side; or worse, such
validation accidentally missing.)

However, some clients will want to bypass such checks (and do it
without risking undefined behavior, namely by ensuring the method preconditions
hold via local invariants in their container type).

For these clients, the `Allocator` trait provides
["unchecked" variants][unchecked variants] of nearly all of its
methods; so `a.alloc_unchecked(layout)` will return an `Option<Address>`
(where `None` corresponds to allocation failure).

The idea here is that `Allocator` implementors are encouraged
to streamline the implmentations of such methods by assuming that all
of the preconditions hold.

 * However, to ease initial `impl Allocator` development for a given
   type, all of the unchecked methods have default implementations
   that call out to their checked counterparts.

 * (In other words, "unchecked" is in some sense a privilege being
   offered to impl's; but there is no guarantee that an arbitrary impl
   takes advantage of the privilege.)

## Object-oriented Allocators

Finally, we get to object-oriented programming.

In general, we expect allocator-parametric code to opt *not* to use
trait objects to generalize over allocators, but instead to use
generic types and instantiate those types with specific concrete
allocators.

Nonetheless, it *is* an option to write `Box<Allocator>` or `&Allocator`.

 * (The allocator methods that are not object-safe, like
   `fn alloc_one<T>(&mut self)`, have a clause `where Self: Sized` to
   ensure that their presence does not cause the `Allocator` trait as
   a whole to become non-object-safe.)


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
  capture a common pattern, but are "less safe" in that they put more
  of an onus on the caller to validate the input parameters before
  calling the methods.

* The `alloc_excess` and `realloc_excess` methods provide a way for
  callers who can make use of excess memory to avoid unnecessary calls
  to `realloc`.

### Why the `Layout` abstraction?

While we do want to require clients to hand the allocator the size and
alignment, we have found that the code to compute such things follows
regular patterns. It makes more sense to factor those patterns out
into a common abstraction; this is what `Layout` provides: a high-level
API for describing the memory layout of a composite structure by
composing the layout of its subparts.

### Why return `Result` rather than a raw pointer?

My hypothesis is that the standard allocator API should embrace
`Result` as the standard way for describing local error conditions in
Rust.

 * A previous version of this RFC attempted to ensure that the use of
   the `Result` type could avoid any additional overhead over a raw
   pointer return value, by using a `NonZero` address type and a
   zero-sized error type attached to the trait via an associated
   `Error` type. But during the RFC process we decided that this
   was not necessary.

### Why return `Result` rather than directly `oom` on failure

Again, my hypothesis is that the standard allocator API should embrace
`Result` as the standard way for describing local error conditions in
Rust.

I want to leave it up to the clients to decide if they can respond to
out-of-memory (OOM) conditions on allocation failure.

However, since I also suspect that some programs would benefit from
contextual information about *which* allocator is reporting memory
exhaustion, I have made `oom` a method of the `Allocator` trait, so
that allocator clients have the option of calling that on error.

### Why is `usable_size` ever needed? Why not call `layout.size()` directly, as is done in the default implementation?

`layout.size()` returns the minimum required size that the client needs.
In a block-based allocator, this may be less than the *actual* size
that the allocator would ever provide to satisfy that kind of
request. Therefore, `usable_size` provides a way for clients to
observe what the minimum actual size of an allocated block for
that`layout` would be, for a given allocator.

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

### The inspiration for Layout

Some aspects of the design of this RFC were selected in the hopes that
it would make such integration easier. In particular, the introduction
of the relatively high-level `Kind` abstraction was developed, in
part, as a way that a GC-aware allocator would build up a tracing
method associated with a layout.

Then I realized that the `Kind` abstraction may be valuable on its
own, without GC: It encapsulates important patterns when working with
representing data as memory records.

(Later we decided to rename `Kind` to `Layout`, in part to avoid
confusion with the use of the word "kind" in the context of
higher-kinded types (HKT).)

So, this RFC offers the `Layout` abstraction without promising that it
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
provide the necessary reflective hooks that a GC would need to access.

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
`Layout` system described here with respect to GC integration.

As far as I know, it is the closest thing we have to a workable system
for allowing client code of allocators to accurately describe the
layout of values they are planning to allocate, which is the main
ingredient I believe to be necessary for the kind of dynamic
reflection that a GC will require of a user-defined allocator.

## Make `Layout` an associated type of `Allocator` trait

I explored making an `AllocLayout` bound and then having

```rust
pub unsafe trait Allocator {
    /// Describes the sort of records that this allocator can
    /// construct.
    type Layout: AllocLayout;

    ...
}
```

Such a design might indeed be workable. (I found it awkward, which is
why I abandoned it.)

But the question is: What benefit does it bring?

The main one I could imagine is that it might allow us to introduce a
division, at the type-system level, between two kinds of allocators:
those that are integrated with the GC (i.e., have an associated
`Allocator::Layout` that ensures that all allocated blocks are scannable
by a GC) and allocators that are *not* integrated with the GC (i.e.,
have an associated `Allocator::Layout` that makes no guarantees about
one will know how to scan the allocated blocks.

However, no such design has proven itself to be "obviously feasible to
implement," and therefore it would be unreasonable to make the `Layout`
an associated type of the `Allocator` trait without having at least a
few motivating examples that *are* clearly feasible and useful.

## Variations on the `Layout` API

 * Should `Layout` offer a `fn resize(&self, new_size: usize) -> Layout` constructor method?
   (Such a method would rule out deriving GC tracers from layouts; but we could
    maybe provide it as an `unsafe` method.)

 * Should `Layout` ensure an invariant that its associated size is
   always a multiple of its alignment?

   * Doing this would allow simplifying a small part of the API,
     namely the distinct `Layout::repeat` (returns both a layout and an
     offset) versus `Layout::array` (where the offset is derivable from
     the input `T`).

   * Such a constraint would have precendent; in particular, the
     `aligned_alloc` function of C11 requires the given size
     be a multiple of the alignment.

   * On the other hand, both the system and jemalloc allocators seem
     to support more flexible allocation patterns. Imposing the above
     invariant implies a certain loss of expressiveness over what we
     already provide today.

 * Should `Layout` ensure an invariant that its associated size is always positive?

   * Pro: Removes something that allocators would need to check about
     input layouts (the backing memory allocators will tend to require
     that the input sizes are positive).

   * Con: Requiring positive size means that zero-sized types do not have an associated
     `Layout`. That's not the end of the world, but it does make the `Layout` API slightly
     less convenient (e.g. one cannot use `extend` with a zero-sized layout to
     forcibly inject padding, because zero-sized layouts do not exist).

 * Should `Layout::align_to` add padding to the associated size? (Probably not; this would
   make it impossible to express certain kinds of patteerns.)

 * Should the `Layout` methods that might "fail" return `Result` instead of `Option`?

## Variations on the `Allocator` API

 * Should the allocator methods take `&self` or `self` rather than `&mut self`.

   As noted during in the RFC comments, nearly every trait goes through a bit
   of an identity crisis in terms of deciding what kind of `self` parameter is
   appropriate.

   The justification for `&mut self` is this:

   * It does not restrict allocator implementors from making sharable allocators:
     to do so, just do `impl<'a> Allocator for &'a MySharedAlloc`, as illustrated
     in the `DumbBumpPool` example.

   * `&mut self` is better than `&self` for simple allocators that are *not* sharable.
     `&mut self` ensures that the allocation methods have exclusive
     access to the underlying allocator state, without resorting to a
     lock. (Another way of looking at it: It moves the onus of using a
     lock outward, to the allocator clients.)

   * One might think that the points made
     above apply equally well to `self` (i.e., if you want to implement an allocator
     that wants to take itself via a `&mut`-reference when the methods take `self`,
     then do `impl<'a> Allocator for &'a mut MyUniqueAlloc`).

     However, the problem with `self` is that if you want to use an
     allocator for *more than one* allocation, you will need to call
     `clone()` (or make the allocator parameter implement
     `Copy`). This means in practice all allocators will need to
     support `Clone` (and thus support sharing in general, as
     discussed in the [Allocators and lifetimes][lifetimes] section).

     (Remember, I'm thinking about allocator-parametric code like
      `Vec<T, A:Allocator>`, which does not know if the `A` is a
      `&mut`-reference. In that context, therefore one cannot assume
      that reborrowing machinery is available to the client code.)

     Put more simply, requiring that allocators implement `Clone` means
     that it will *not* be pratical to do
     `impl<'a> Allocator for &'a mut MyUniqueAlloc`.

     By using `&mut self` for the allocation methods, we can encode
     the expected use case of an *unshared* allocator that is used
     repeatedly in a linear fashion (e.g. vector that needs to
     reallocate its backing storage).

 * Should the types representing allocated storage have lifetimes attached?
   (E.g. `fn alloc<'a>(&mut self, layout: &alloc::Layout) -> Address<'a>`.)

   I think Gankro [put it best](https://github.com/rust-lang/rfcs/pull/1398#issuecomment-164003160):

   > This is a low-level unsafe interface, and the expected usecases make it
   > both quite easy to avoid misuse, and impossible to use lifetimes
   > (you want a struct to store the allocator and the allocated elements).
   > Any time we've tried to shove more lifetimes into these kinds of
   > interfaces have just been an annoying nuisance necessitating
   > copy-lifetime/transmute nonsense.

 * Should `Allocator::alloc` be safe instead of `unsafe fn`?
 
   * Clearly `fn dealloc` and `fn realloc` need to be `unsafe`, since
     feeding in improper inputs could cause unsound behavior. But is
     there any analogous input to `fn alloc` that could cause
     unsoundness (assuming that the `Layout` struct enforces invariants
     like "the associated size is non-zero")?

   * (I left it as `unsafe fn alloc` just to keep the API uniform with
     `dealloc` and `realloc`.)

 * Should `Allocator::realloc` not require that `new_layout.align()`
   evenly divide `layout.align()`? In particular, it is not too
   expensive to check if the two layouts are not compatible, and fall
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

 * Should the precondition of allocation methods be loosened to
   accept zero-sized types?

   Right now, there is a requirement that the allocation requests
   denote non-zero sized types (this requirement is encoded in two
   ways: for `Layout`-consuming methods like `alloc`, it is enforced
   via the invariant that the `Size` is a `NonZero`, and this is
   enforced by checks in the `Layout` construction code; for the
   convenience methods like `alloc_one`, they will return `Err` if the
   allocation request is zero-sized).

   The main motivation for this restriction is some underlying system
   allocators, like `jemalloc`, explicitly disallow zero-sized
   inputs. Therefore, to remove all unnecessary control-flow branches
   between the client and the underlying allocator, the `Allocator`
   trait is bubbling that restriction up and imposing it onto the
   clients, who will presumably enforce this invariant via
   container-specific means.

   But: pre-existing container types (like `Vec<T>`) already
   *allow* zero-sized `T`. Therefore, there is an unfortunate mismatch
   between the ideal API those container would prefer for their
   allocators and the actual service that this `Allocator` trait is
   providing.

   So: Should we lift this precondition of the allocation methods, and allow
   zero-sized requests (which might be handled by a global sentinel value, or
   by an allocator-specific sentinel value, or via some other means -- this
   would have to be specified as part of the Allocator API)?

   (As a middle ground, we could lift the precondition solely for the convenience
   methods like `fn alloc_one` and `fn alloc_array`; that way, the most low-level
   methods like `fn alloc` would continue to minimize the overhead they add
   over the underlying system allocator, while the convenience methods would truly
   be convenient.)

 * Should `oom` be a free-function rather than a method on `Allocator`?
   (The reason I want it on `Allocator` is so that it can provide feedback
   about the allocator's state at the time of the OOM. Zoxc has argued
   on the RFC thread that some forms of static analysis, to prove `oom` is
   never invoked, would prefer it to be a free function.)

# Unresolved questions
[unresolved]: #unresolved-questions

 * Since we cannot do `RefCell<Pool>` (see FIXME above), what is
   our standard recommendation for what to do instead?

 * Should `Layout` be an associated type of `Allocator` (see
   [alternatives][] section for discussion).
   (In fact, most of the "Variations correspond to potentially
   unresolved questions.)

 * Are the type definitions for `Size`, `Capacity`, `Alignment`, and
   `Address` an abuse of the `NonZero` type? (Or do we just need some
   constructor for `NonZero` that asserts that the input is non-zero)?

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

 * Should `Allocator::oom` also take a `std::fmt::Arguments<'a>` parameter
   so that clients can feed in context-specific information that is not
   part of the original input `Layout` argument? (I have not done this
   mainly because I do not want to introduce a dependency on `libstd`.)

# Change History

* Changed `fn usable_size` to return `(l, m)` rathern than just `m`.

* Removed `fn is_transient` from `trait AllocError`, and removed discussion
  of transient errors from the API.

* Made `fn dealloc` method infallible (i.e. removed its `Result` return type).

* Alpha-renamed `alloc::Kind` type to `alloc::Layout`, and made it non-`Copy`.

* Revised `fn oom` method to take the `Self::Error` as an input (so that the
  allocator can, indirectly, feed itself information about what went wrong).

* Removed associated `Error` type from `Allocator` trait; all methods now use `AllocErr`
  for error type. Removed `AllocError` trait and `MemoryExhausted` error.

* Removed `fn max_size` and `fn max_align` methods; we can put them back later if
  someone demonstrates a need for them.

* Added `fn realloc_in_place`.

* Removed uses of `NonZero`. Made `Layout` able to represent zero-sized layouts.
  A given `Allocator` may or may not support zero-sized layouts.

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
use core::mem;
use core::nonzero::NonZero;
use core::ptr::{self, Unique};

```

### Type Aliases
[type aliases]: #type-aliases

```rust
pub type Size = usize;
pub type Capacity = usize;
pub type Alignment = usize;

pub type Address = *mut u8;

/// Represents the combination of a starting address and
/// a total capacity of the returned block.
pub struct Excess(Address, Capacity);

fn size_align<T>() -> (usize, usize) {
    (mem::size_of::<T>(), mem::align_of::<T>())
}

```

### Layout API
[layout api]: #layout-api

```rust
/// Category for a memory record.
///
/// An instance of `Layout` describes a particular layout of memory.
/// You build a `Layout` up as an input to give to an allocator.
///
/// All layouts have an associated non-negative size and positive alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    // size of the requested block of memory, measured in bytes.
    size: Size,
    // alignment of the requested block of memory, measured in bytes.
    // we ensure that this is always a power-of-two, because API's
    ///like `posix_memalign` require it and it is a reasonable
    // constraint to impose on Layout constructors.
    //
    // (However, we do not analogously require `align >= sizeof(void*)`,
    //  even though that is *also* a requirement of `posix_memalign`.)
    align: Alignment,
}


// FIXME: audit default implementations for overflow errors,
// (potentially switching to overflowing_add and
//  overflowing_mul as necessary).

impl Layout {
    // (private constructor)
    fn from_size_align(size: usize, align: usize) -> Layout {
        assert!(align.is_power_of_two());
        assert!(align > 0);
        Layout { size: size, align: align }
    }

    /// The minimum size in bytes for a memory block of this layout.
    pub fn size(&self) -> usize { self.size }

    /// The minimum byte alignment for a memory block of this layout.
    pub fn align(&self) -> usize { self.align }

    /// Constructs a `Layout` suitable for holding a value of type `T`.
    pub fn new<T>() -> Self {
        let (size, align) = size_align::<T>();
        Layout::from_size_align(size, align)
    }

    /// Produces layout describing a record that could be used to
    /// allocate backing structure for `T` (which could be a trait
    /// or other unsized type like a slice).
    pub fn for_value<T: ?Sized>(t: &T) -> Self {
        let (size, align) = (mem::size_of_val(t), mem::align_of_val(t));
        Layout::from_size_align(size, align)
    }

    /// Creates a layout describing the record that can hold a value
    /// of the same layout as `self`, but that also is aligned to
    /// alignment `align` (measured in bytes).
    ///
    /// If `self` already meets the prescribed alignment, then returns
    /// `self`.
    ///
    /// Note that this method does not add any padding to the overall
    /// size, regardless of whether the returned layout has a different
    /// alignment. In other words, if `K` has size 16, `K.align_to(32)`
    /// will *still* have size 16.
    pub fn align_to(&self, align: Alignment) -> Self {
        if align > self.align {
            let pow2_align = align.checked_next_power_of_two().unwrap();
            debug_assert!(pow2_align > 0); // (this follows from self.align > 0...)
            Layout { align: pow2_align,
                     ..*self }
        } else {
            self.clone()
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
        debug_assert!(align <= self.align());
        let len = self.size();
        let len_rounded_up = (len + align - 1) & !(align - 1);
        return len_rounded_up - len;
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each to
    /// ensure that each instance is given its requested size and
    /// alignment. On success, returns `(k, offs)` where `k` is the
    /// layout of the array and `offs` is the distance between the start
    /// of each element in the array.
    ///
    /// On arithmetic overflow, returns `None`.
    pub fn repeat(&self, n: usize) -> Option<(Self, usize)> {
        let padded_size = match self.size.checked_add(self.padding_needed_for(self.align)) {
            None => return None,
            Some(padded_size) => padded_size,
        };
        let alloc_size = match padded_size.checked_mul(n) {
            None => return None,
            Some(alloc_size) => alloc_size,
        };
        Some((Layout::from_size_align(alloc_size, self.align), padded_size))
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next`, including any necessary padding to ensure that `next`
    /// will be properly aligned. Note that the result layout will
    /// satisfy the alignment properties of both `self` and `next`.
    ///
    /// Returns `Some((k, offset))`, where `k` is layout of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// On arithmetic overflow, returns `None`.
    pub fn extend(&self, next: Self) -> Option<(Self, usize)> {
        let new_align = cmp::max(self.align, next.align);
        let realigned = Layout { align: new_align, ..*self };
        let pad = realigned.padding_needed_for(new_align);
        let offset = self.size() + pad;
        let new_size = offset + next.size();
        Some((Layout::from_size_align(new_size, new_align), offset))
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with no padding between each instance.
    ///
    /// On arithmetic overflow, returns `None`.
    pub fn repeat_packed(&self, n: usize) -> Option<Self> {
        let scaled = match self.size().checked_mul(n) {
            None => return None,
            Some(scaled) => scaled,
        };
        let size = { assert!(scaled > 0); scaled };
        Some(Layout { size: size, align: self.align })
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incoporated *at all* into the resulting layout.
    ///
    /// Returns `(k, offset)`, where `k` is layout of the concatenated
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
        let new_size = match self.size().checked_add(next.size()) {
            None => return None,
            Some(new_size) => new_size,
        };
        Some((Layout { size: new_size, ..*self }, self.size()))
    }

    // Below family of methods *assume* inputs are pre- or
    // post-validated in some manner. (The implementations here
    ///do indirectly validate, but that is not part of their
    /// specification.)
    //
    // Since invalid inputs could yield ill-formed layouts, these
    // methods are `unsafe`.

    /// Creates layout describing the record for a single instance of `T`.
    pub unsafe fn new_unchecked<T>() -> Self {
        let (size, align) = size_align::<T>();
        Layout::from_size_align(size, align)
    }


    /// Creates a layout describing the record for `self` followed by
    /// `next`, including any necessary padding to ensure that `next`
    /// will be properly aligned. Note that the result layout will
    /// satisfy the alignment properties of both `self` and `next`.
    ///
    /// Returns `(k, offset)`, where `k` is layout of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded witnin the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// Requires no arithmetic overflow from inputs.
    pub unsafe fn extend_unchecked(&self, next: Self) -> (Self, usize) {
        self.extend(next).unwrap()
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each.
    ///
    /// Requires non-zero `n` and no arithmetic overflow from inputs.
    /// (See also the `fn array` checked variant.)
    pub unsafe fn repeat_unchecked(&self, n: usize) -> (Self, usize) {
        self.repeat(n).unwrap()
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with no padding between each instance.
    ///
    /// Requires non-zero `n` and no arithmetic overflow from inputs.
    /// (See also the `fn array_packed` checked variant.)
    pub unsafe fn repeat_packed_unchecked(&self, n: usize) -> Self {
        self.repeat_packed(n).unwrap()
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incoporated *at all* into the resulting layout.
    ///
    /// Returns `(k, offset)`, where `k` is layout of the concatenated
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

    /// Creates a layout describing the record for a `[T; n]`.
    ///
    /// On zero `n`, zero-sized `T`, or arithmetic overflow, returns `None`.
    pub fn array<T>(n: usize) -> Option<Self> {
        Layout::new::<T>()
            .repeat(n)
            .map(|(k, offs)| {
                debug_assert!(offs == mem::size_of::<T>());
                k
            })
    }

    /// Creates a layout describing the record for a `[T; n]`.
    ///
    /// Requires nonzero `n`, nonzero-sized `T`, and no arithmetic
    /// overflow; otherwise behavior undefined.
    pub fn array_unchecked<T>(n: usize) -> Self {
        Layout::array::<T>(n).unwrap()
    }

}

```

### AllocErr API
[error api]: #allocerr-api

```rust
/// The `AllocErr` error specifies whether an allocation failure is
/// specifically due to resource exhaustion or if it is due to
/// something wrong when combining the given input arguments with this
/// allocator.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AllocErr {
    /// Error due to hitting some resource limit or otherwise running
    /// out of memory. This condition strongly implies that *some*
    /// series of deallocations would allow a subsequent reissuing of
    /// the original allocation request to succeed.
    Exhausted { request: Layout },

    /// Error due to allocator being fundamentally incapable of
    /// satisfying the original request. This condition implies that
    /// such an allocation request will never succeed on the given
    /// allocator, regardless of environment, memory pressure, or
    /// other contextual condtions.
    ///
    /// For example, an allocator that does not support zero-sized
    /// blocks can return this error variant.
    Unsupported { details: &'static str },
}

impl AllocErr {
    pub fn invalid_input(details: &'static str) -> Self {
        AllocErr::Unsupported { details: details }
    }
    pub fn is_memory_exhausted(&self) -> bool {
        if let AllocErr::Exhausted { .. } = *self { true } else { false }
    }
    pub fn is_request_unsupported(&self) -> bool {
        if let AllocErr::Unsupported { .. } = *self { true } else { false }
    }
}

/// The `CannotReallocInPlace` error is used when `fn realloc_in_place`
/// was unable to reuse the given memory block for a requested layout.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CannotReallocInPlace;

```

### Allocator trait header
[trait header]: #allocator-trait-header

```rust
/// An implementation of `Allocator` can allocate, reallocate, and
/// deallocate arbitrary blocks of data described via `Layout`.
///
/// Some of the methods require that a layout *fit* a memory block.
/// What it means for a layout to "fit" a memory block means is that
/// the following two conditions must hold:
///
/// 1. The block's starting address must be aligned to `layout.align()`.
///
/// 2. The block's size must fall in the range `[use_min, use_max]`, where:
///
///    * `use_min` is `self.usable_size(layout).0`, and
///
///    * `use_max` is the capacity that was (or would have been)
///      returned when (if) the block was allocated via a call to
///      `alloc_excess` or `realloc_excess`.
///
/// Note that:
///
///  * the size of the layout most recently used to allocate the block
///    is guaranteed to be in the range `[use_min, use_max]`, and
///
///  * a lower-bound on `use_max` can be safely approximated by a call to
///    `usable_size`.
///
pub unsafe trait Allocator {

```

### Allocator core alloc and dealloc
[alloc and dealloc]: #allocator-core-alloc-and-dealloc

```rust
    /// Returns a pointer suitable for holding data described by
    /// `layout`, meeting its size and alignment guarantees.
    ///
    /// The returned block of storage may or may not have its contents
    /// initialized. (Extension subtraits might restrict this
    /// behavior, e.g. to ensure initialization.)
    ///
    /// Returning `Err` indicates that either memory is exhausted or `layout` does
    /// not meet allocator's size or alignment constraints.
    ///
    /// Implementations are encouraged to return `Err` on memory
    /// exhaustion rather than panicking or aborting, but this is
    /// not a strict requirement. (Specifically: it is *legal* to use
    /// this trait to wrap an underlying native allocation library
    /// that aborts on memory exhaustion.)
    unsafe fn alloc(&mut self, layout: Layout) -> Result<Address, AllocErr>;

    /// Deallocate the memory referenced by `ptr`.
    ///
    /// `ptr` must have previously been provided via this allocator,
    /// and `layout` must *fit* the provided block (see above);
    /// otherwise yields undefined behavior.
    unsafe fn dealloc(&mut self, ptr: Address, layout: Layout);

    /// Allocator-specific method for signalling an out-of-memory
    /// condition.
    ///
    /// Implementations of the `oom` method are discouraged from
    /// infinitely regressing in nested calls to `oom`. In
    /// practice this means implementors should eschew allocating,
    /// especially from `self` (directly or indirectly).
    ///
    /// Implementions of this trait's allocation methods are discouraged
    /// from panicking (or aborting) in the event of memory exhaustion;
    /// instead they should return an appropriate error from the
    /// invoked method, and let the client decide whether to invoke
    /// this `oom` method.
    fn oom(&mut self, _: AllocErr) -> ! {
        unsafe { ::core::intrinsics::abort() }
    }
```

### Allocator-specific quantities and limits
[quantites and limits]: #allocator-specific-quantities-and-limits

```rust
    // == ALLOCATOR-SPECIFIC QUANTITIES AND LIMITS ==
    // usable_size

    /// Returns bounds on the guaranteed usable size of a successful
    /// allocation created with the specified `layout`.
    ///
    /// In particular, for a given layout `k`, if `usable_size(k)` returns
    /// `(l, m)`, then one can use a block of layout `k` as if it has any
    /// size in the range `[l, m]` (inclusive).
    ///
    /// (All implementors of `fn usable_size` must ensure that
    /// `l <= k.size() <= m`)
    ///
    /// Both the lower- and upper-bounds (`l` and `m` respectively) are
    /// provided: An allocator based on size classes could misbehave
    /// if one attempts to deallocate a block without providing a
    /// correct value for its size (i.e., one within the range `[l, m]`).
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
    unsafe fn usable_size(&self, layout: &Layout) -> (Capacity, Capacity) {
        (layout.size(), layout.size())
    }

```

### Allocator methods for memory reuse
[memory reuse]: #allocator-methods-for-memory-reuse

```rust
    // == METHODS FOR MEMORY REUSE ==
    // realloc. alloc_excess, realloc_excess
    
    /// Returns a pointer suitable for holding data described by
    /// `new_layout`, meeting its size and alignment guarantees. To
    /// accomplish this, this may extend or shrink the allocation
    /// referenced by `ptr` to fit `new_layout`.
    ///
    /// * `ptr` must have previously been provided via this allocator.
    ///
    /// * `layout` must *fit* the `ptr` (see above). (The `new_layout`
    ///   argument need not fit it.)
    ///
    /// Behavior undefined if either of latter two constraints are unmet.
    ///
    /// In addition, `new_layout` should not impose a different alignment
    /// constraint than `layout`. (In other words, `new_layout.align()`
    /// should equal `layout.align()`.)
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
    /// Returns `Err` only if `new_layout` does not meet the allocator's
    /// size and alignment constraints of the allocator or the
    /// alignment of `layout`, or if reallocation otherwise fails. (Note
    /// that did not say "if and only if" -- in particular, an
    /// implementation of this method *can* return `Ok` if
    /// `new_layout.align() != old_layout.align()`; or it can return `Err`
    /// in that scenario, depending on whether this allocator
    /// can dynamically adjust the alignment constraint for the block.)
    ///
    /// If this method returns `Err`, then ownership of the memory
    /// block has not been transferred to this allocator, and the
    /// contents of the memory block are unaltered.
    unsafe fn realloc(&mut self,
                      ptr: Address,
                      layout: Layout,
                      new_layout: Layout) -> Result<Address, AllocErr> {
        let (min, max) = self.usable_size(&layout);
        let s = new_layout.size();
        // All Layout alignments are powers of two, so a comparison
        // suffices here (rather than resorting to a `%` operation).
        if min <= s && s <= max && new_layout.align() <= layout.align() {
            return Ok(ptr);
        } else {
            let new_size = new_layout.size();
            let old_size = layout.size();
            let result = self.alloc(new_layout);
            if let Ok(new_ptr) = result {
                ptr::copy(ptr as *const u8, new_ptr, cmp::min(old_size, new_size));
                self.dealloc(ptr, layout);
            }
            result
        }
    }

    /// Behaves like `fn alloc`, but also returns the whole size of
    /// the returned block. For some `layout` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        let usable_size = self.usable_size(&layout);
        self.alloc(layout).map(|p| Excess(p, usable_size.1))
    }

    /// Behaves like `fn realloc`, but also returns the whole size of
    /// the returned block. For some `layout` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    unsafe fn realloc_excess(&mut self,
                             ptr: Address,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr> {
        let usable_size = self.usable_size(&new_layout);
        self.realloc(ptr, layout, new_layout)
            .map(|p| Excess(p, usable_size.1))
    }

    /// Attempts to extend the allocation referenced by `ptr` to fit `new_layout`.
    ///
    /// * `ptr` must have previously been provided via this allocator.
    ///
    /// * `layout` must *fit* the `ptr` (see above). (The `new_layout`
    ///   argument need not fit it.)
    ///
    /// Behavior undefined if either of latter two constraints are unmet.
    ///
    /// If this returns `Ok`, then the allocator has asserted that the
    /// memory block referenced by `ptr` now fits `new_layout`, and thus can
    /// be used to carry data of that layout. (The allocator is allowed to
    /// expend effort to accomplish this, such as extending the memory block to
    /// include successor blocks, or virtual memory tricks.)
    ///
    /// If this returns `Err`, then the allocator has made no assertion
    /// about whether the memory block referenced by `ptr` can or cannot
    /// fit `new_layout`.
    ///
    /// In either case, ownership of the memory block referenced by `ptr`
    /// has not been transferred, and the contents of the memory block
    /// are unaltered.
    unsafe fn realloc_in_place(&mut self,
                               ptr: Address,
                               layout: Layout,
                               new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        let (_, _, _) = (ptr, layout, new_layout);
        Err(CannotReallocInPlace)
    }
```

### Allocator convenience methods for common usage patterns
[common usage patterns]: #allocator-convenience-methods-for-common-usage-patterns

```rust
    // == COMMON USAGE PATTERNS ==
    // alloc_one, dealloc_one, alloc_array, realloc_array. dealloc_array
    
    /// Allocates a block suitable for holding an instance of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `alloc`/`realloc` methods of this allocator.
    ///
    /// May return `Err` for zero-sized `T`.
    unsafe fn alloc_one<T>(&mut self) -> Result<Unique<T>, AllocErr>
        where Self: Sized {
        let k = Layout::new::<T>();
        if k.size() > 0 {
            self.alloc(k).map(|p|Unique::new(*p as *mut T))
        } else {
            Err(AllocErr::invalid_input("zero-sized type invalid for alloc_one"))
        }
    }

    /// Deallocates a block suitable for holding an instance of `T`.
    ///
    /// The given block must have been produced by this allocator,
    /// and must be suitable for storing a `T` (in terms of alignment
    /// as well as minimum and maximum size); otherwise yields
    /// undefined behavior.
    ///
    /// Captures a common usage pattern for allocators.
    unsafe fn dealloc_one<T>(&mut self, mut ptr: Unique<T>)
        where Self: Sized {
        let raw_ptr = ptr.get_mut() as *mut T as *mut u8;
        self.dealloc(raw_ptr, Layout::new::<T>());
    }

    /// Allocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `alloc`/`realloc` methods of this allocator.
    ///
    /// May return `Err` for zero-sized `T` or `n == 0`.
    ///
    /// Always returns `Err` on arithmetic overflow.
    unsafe fn alloc_array<T>(&mut self, n: usize) -> Result<Unique<T>, AllocErr>
        where Self: Sized {
        match Layout::array::<T>(n) {
            Some(ref layout) if layout.size() > 0 => {
                self.alloc(layout.clone())
                    .map(|p| {
                        println!("alloc_array layout: {:?} yielded p: {:?}", layout, p);
                        Unique::new(p as *mut T)
                    })
            }
            _ => Err(AllocErr::invalid_input("invalid layout for alloc_array")),
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
    ///
    /// May return `Err` for zero-sized `T` or `n == 0`.
    ///
    /// Always returns `Err` on arithmetic overflow.
    unsafe fn realloc_array<T>(&mut self,
                               ptr: Unique<T>,
                               n_old: usize,
                               n_new: usize) -> Result<Unique<T>, AllocErr>
        where Self: Sized {
        match (Layout::array::<T>(n_old), Layout::array::<T>(n_new), *ptr) {
            (Some(ref k_old), Some(ref k_new), ptr) if k_old.size() > 0 && k_new.size() > 0 => {
                self.realloc(ptr as *mut u8, k_old.clone(), k_new.clone())
                    .map(|p|Unique::new(p as *mut T))
            }
            _ => {
                Err(AllocErr::invalid_input("invalid layout for realloc_array"))
            }
        }
    }

    /// Deallocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    unsafe fn dealloc_array<T>(&mut self, ptr: Unique<T>, n: usize) -> Result<(), AllocErr>
        where Self: Sized {
        let raw_ptr = *ptr as *mut u8;
        match Layout::array::<T>(n) {
            Some(ref k) if k.size() > 0 => {
                Ok(self.dealloc(raw_ptr, k.clone()))
            }
            _ => {
                Err(AllocErr::invalid_input("invalid layout for dealloc_array"))
            }
        }
    }

```

### Allocator unchecked method variants
[unchecked variants]: #allocator-unchecked-method-variants

```rust
    // UNCHECKED METHOD VARIANTS

    /// Returns a pointer suitable for holding data described by
    /// `layout`, meeting its size and alignment guarantees.
    ///
    /// The returned block of storage may or may not have its contents
    /// initialized. (Extension subtraits might restrict this
    /// behavior, e.g. to ensure initialization.)
    ///
    /// Returns `None` if request unsatisfied.
    ///
    /// Behavior undefined if input does not meet size or alignment
    /// constraints of this allocator.
    unsafe fn alloc_unchecked(&mut self, layout: Layout) -> Option<Address> {
        // (default implementation carries checks, but impl's are free to omit them.)
        self.alloc(layout).ok()
    }

    /// Returns a pointer suitable for holding data described by
    /// `new_layout`, meeting its size and alignment guarantees. To
    /// accomplish this, may extend or shrink the allocation
    /// referenced by `ptr` to fit `new_layout`.
    ////
    /// (In other words, ownership of the memory block associated with
    /// `ptr` is first transferred back to this allocator, but the
    /// same block may or may not be transferred back as the result of
    /// this call.)
    ///
    /// * `ptr` must have previously been provided via this allocator.
    ///
    /// * `layout` must *fit* the `ptr` (see above). (The `new_layout`
    ///   argument need not fit it.)
    ///
    /// * `new_layout` must meet the allocator's size and alignment
    ///    constraints. In addition, `new_layout.align()` must equal
    ///    `layout.align()`. (Note that this is a stronger constraint
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
                                layout: Layout,
                                new_layout: Layout) -> Option<Address> {
        // (default implementation carries checks, but impl's are free to omit them.)
        self.realloc(ptr, layout, new_layout).ok()
    }

    /// Behaves like `fn alloc_unchecked`, but also returns the whole
    /// size of the returned block. 
    unsafe fn alloc_excess_unchecked(&mut self, layout: Layout) -> Option<Excess> {
        self.alloc_excess(layout).ok()
    }

    /// Behaves like `fn realloc_unchecked`, but also returns the
    /// whole size of the returned block.
    unsafe fn realloc_excess_unchecked(&mut self,
                                       ptr: Address,
                                       layout: Layout,
                                       new_layout: Layout) -> Option<Excess> {
        self.realloc_excess(ptr, layout, new_layout).ok()
    }


    /// Allocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// Requires inputs are non-zero and do not cause arithmetic
    /// overflow, and `T` is not zero sized; otherwise yields
    /// undefined behavior.
    unsafe fn alloc_array_unchecked<T>(&mut self, n: usize) -> Option<Unique<T>>
        where Self: Sized {
        let layout = Layout::array_unchecked::<T>(n);
        self.alloc_unchecked(layout).map(|p|Unique::new(*p as *mut T))
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
                                         n_new: usize) -> Option<Unique<T>>
        where Self: Sized {
        let (k_old, k_new, ptr) = (Layout::array_unchecked::<T>(n_old),
                                   Layout::array_unchecked::<T>(n_new),
                                   *ptr);
        self.realloc_unchecked(ptr as *mut u8, k_old, k_new)
            .map(|p|Unique::new(*p as *mut T))
    }

    /// Deallocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// Requires inputs are non-zero and do not cause arithmetic
    /// overflow, and `T` is not zero sized; otherwise yields
    /// undefined behavior.
    unsafe fn dealloc_array_unchecked<T>(&mut self, ptr: Unique<T>, n: usize)
        where Self: Sized {
        let layout = Layout::array_unchecked::<T>(n);
        self.dealloc(*ptr as *mut u8, layout);
    }
}
```
