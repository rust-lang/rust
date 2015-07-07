% Atomics

Rust pretty blatantly just inherits C11's memory model for atomics. This is not
due this model being particularly excellent or easy to understand. Indeed, this
model is quite complex and known to have [several flaws][C11-busted]. Rather,
it is a pragmatic concession to the fact that *everyone* is pretty bad at modeling
atomics. At very least, we can benefit from existing tooling and research around
C.

Trying to fully explain the model is fairly hopeless. If you want all the
nitty-gritty details, you should check out [C's specification][C11-model].
Still, we'll try to cover the basics and some of the problems Rust developers
face.

The C11 memory model is fundamentally about trying to bridge the gap between C's
single-threaded semantics, common compiler optimizations, and hardware peculiarities
in the face of a multi-threaded environment. It does this by splitting memory
accesses into two worlds: data accesses, and atomic accesses.

Data accesses are the bread-and-butter of the programming world. They are
fundamentally unsynchronized and compilers are free to aggressively optimize
them. In particular data accesses are free to be reordered by the compiler
on the assumption that the program is single-threaded. The hardware is also free
to propagate the changes made in data accesses as lazily and inconsistently as
it wants to other threads. Mostly critically, data accesses are where we get data
races. These are pretty clearly awful semantics to try to write a multi-threaded
program with.

Atomic accesses are the answer to this. Each atomic access can be marked with
an *ordering*. The set of orderings Rust exposes are:

* Sequentially Consistent (SeqCst)
* Release
* Acquire
* Relaxed

(Note: We explicitly do not expose the C11 *consume* ordering)

TODO: give simple "basic" explanation of these
TODO: implementing Arc example (why does Drop need the trailing barrier?)





[C11-busted]: http://plv.mpi-sws.org/c11comp/popl15.pdf
[C11-model]: http://en.cppreference.com/w/c/atomic/memory_order
