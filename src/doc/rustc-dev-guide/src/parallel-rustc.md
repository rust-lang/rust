# Parallel Compilation

Most of the compiler is not parallel. This represents an opportunity for
improving compiler performance.

As of January 2021 <!-- date: 2021-01 -->, work on explicitly parallelizing the
compiler has stalled. There is a lot of design and correctness work that needs
to be done.

One can try out the current parallel compiler work by enabling it in the
`config.toml`.

There are a few basic ideas in this effort:

- There are a lot of loops in the compiler that just iterate over all items in
  a crate. These can possibly be parallelized.
- We can use (a custom fork of) [`rayon`] to run tasks in parallel. The custom
  fork allows the execution of DAGs of tasks, not just trees.
- There are currently a lot of global data structures that need to be made
  thread-safe. A key strategy here has been converting interior-mutable
  data-structures (e.g. `Cell`) into their thread-safe siblings (e.g. `Mutex`).

[`rayon`]: https://crates.io/crates/rayon

As of February 2021 <!-- date: 2021-02 -->, much of this effort is on hold due
to lack of manpower. We have a working prototype with promising performance
gains in many cases. However, there are two blockers:

- It's not clear what invariants need to be upheld that might not hold in the
  face of concurrency. An auditing effort was underway, but seems to have
  stalled at some point.

- There is a lot of lock contention, which actually degrades performance as the
  number of threads increases beyond 4.

Here are some resources that can be used to learn more (note that some of them
are a bit out of date):

- [This IRLO thread by Zoxc, one of the pioneers of the effort][irlo0]
- [This list of interior mutability in the compiler by nikomatsakis][imlist]
- [This IRLO thread by alexchricton about performance][irlo1]
- [This tracking issue][tracking]

[irlo0]: https://internals.rust-lang.org/t/parallelizing-rustc-using-rayon/6606
[imlist]: https://github.com/nikomatsakis/rustc-parallelization/blob/master/interior-mutability-list.md
[irlo1]: https://internals.rust-lang.org/t/help-test-parallel-rustc/11503
[tracking]: https://github.com/rust-lang/rust/issues/48685
