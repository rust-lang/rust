% Example: Implementing Vec

To bring everything together, we're going to write `std::Vec` from scratch.
Because all the best tools for writing unsafe code are unstable, this
project will only work on nightly (as of Rust 1.2.0). With the exception of the
allocator API, much of the unstable code we'll use is expected to be stabilized
in a similar form as it is today.

However we will generally try to avoid unstable code where possible. In
particular we won't use any intrinsics that could make a code a little
bit nicer or efficient because intrinsics are permanently unstable. Although
many intrinsics *do* become stabilized elsewhere (`std::ptr` and `str::mem`
consist of many intrinsics).

Ultimately this means our implementation may not take advantage of all
possible optimizations, though it will be by no means *naive*. We will
definitely get into the weeds over nitty-gritty details, even
when the problem doesn't *really* merit it.

You wanted advanced. We're gonna go advanced.
