# Shadow Memory
In Rust, every pointer is both an address to a memory location and a [*provenance*](https://github.com/rust-lang/rfcs/pull/3559) value.

Provenance is an abstract concept that we use to reason about where a pointer originates from
and what capabilities it has to access memory. It isn't a concrete value floating around as your program executes—at least, not by default—but it's a key part of Rust's formal model of
correctness. Tools like [Miri](https://github.com/rust-lang/miri) (a Rust interpreter) and *BorrowSanitizer* track the provenance metadata for each pointer to detect errors. For example:
```
unsafe {
  let layout = Layout::new::<u8>();
  let ptr = alloc(layout);
  *ptr.offset(2)
}
```
From provenance, we know that `ptr` is pointing at a chunk of memory with size 1, which is located at the address first associated with `ptr` before it is offset. By tracking this information, we know that reading from this location with an offset of 2 is an access out-of-bounds. This is one of the many forms of [undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html) that Rust programmers need to avoid when working with `unsafe` code.

Rust's provenance metadata can be concretely represented as an enum with two values. The following code is used by Miri to represent provenance:
```
pub enum Provenance {
    /// For pointers with concrete provenance. we exactly know which allocation they are attached to
    /// and what their borrow tag is.
    Concrete {
        alloc_id: AllocId,
        /// Borrow Tracker tag.
        tag: BorTag,
    },
    // We'll ignore this for now
    Wildcard
}
```
The `alloc_id` is a unique identifier for an allocation, which we can use to lookup whether the allocation is still valid, what its size and alignment are, and where it's located. The `tag` is a pointer's permission to access that allocation. It is a unique identifier for a node within the "tree" of Rust's Tree Borrows aliasing model. We can use this tag to lookup the pointer's permission and update the state of the tree based on each access that takes place during execution.

Now, we need to associate every pointer with this provenance value. But that's a tricky prospect. Sure, on the stack, we can pass around an extra two words with each pointer for the allocation ID and borrow tag. But what about the heap? If a struct contains a pointer and it's stored in heap memory, then where should that pointer's provenance go? One option is to change the layout of a pointer to be three words long. However, these "fat pointers" will only be compatible with code that we can instrument using our sanitizer. Uninstrumented libraries will only expect to receive normal-sized pointers, and we don't want to force our users to compile *everything* from scratch every time that they want to use BorrowSanitizer. 

Our solution to this problem is *Shadow Memory* 👻. Instead of changing the layout of a pointer, we'll create a separate *shadow heap* and *shadow stack* dedicated to storing provenance values. Your first task will be to take our initial "stub" implementation of the shadow heap and extend it so that can store and return provenance values. 

### Task 1 - Setup
Complete the [Quickstart guide](https://borrowsanitizer.com/quickstart.html) for BorrowSanitizer. While it's compiling, read [the shadow memory paper](https://valgrind.org/docs/shadow-memory2007.pdf). No need to cover all of it yet; just make sure you get to Sections 1 and 3.0-3.4.

Then, open our fork of Rust in an IDE and navigate to the path `src/tools/bsan/bsanrt`. This directory contains the implementation of the BorrowSanitizer runtime. Open the file `src/shadow.rs`, which contains the implementation of the shadow heap. Read through the comments in the source code and compare our implementation to the one shown in Section 3.1 of the shadow memory paper. Send a message over in the [dev channel](https://bsan.zulipchat.com/#narrow/channel/478480-dev) of our Zulip if you have any questions. 

### Task 2 - Malloc
You will be implementing the function `malloc` for our shadow table. This function should have the following signature. 
```
unsafe fn malloc(&mut self, ptr: *mut u8, size: usize) 
```
We will call this function *after* our system allocator returns. It will take as input a pointer to the start of the allocation (`ptr`) and the size of the allocation (`size`). This function should allocate enough `L2` pages to cover the allocation. (e.g. `size / 64kb`). These pages should be zero-initialized. As you're implementing this feature, the quickest way to rebuild *just* the runtime is to run `./x.py build bsanrt`.

After implementing this function, navigate to the bottom and add a few unit tests for it. We should have at least one test that only creates a new `ShadowMap` and allocates a page, and then another that tries to lookup a provenance value within the page. You can run these unit tests using `./x.py test src/tools/bsan/bsanrt`. 

When finished, make sure to run `./x.py fmt`. This is required for any commit to pass continuous integration. Then, push your changes to a new upstream branch and [create a pull request](https://github.com/BorrowSanitizer/rust/pulls). Let me know when you do, and then merge it when CI passes. 

### Task 3 - Stdlib
We need corresponding shadow functions for `free`, `memcpy`, `realloc`, `memmove`, `strcpy`, and any one of the other system calls within the [C standard library](https://en.cppreference.com/w/c/header) that create, destroy, or copy memory. Most would be in [`stdlib.h`](https://en.cppreference.com/w/c/program) and [`string.h`](https://en.cppreference.com/w/c/string/byte). I'd recommend starting with `free` before moving onto the others.