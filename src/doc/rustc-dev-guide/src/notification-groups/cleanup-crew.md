# Cleanup Crew

**Github Label:** [ICEBreaker-Cleanup-Crew]

[ICEBreaker-Cleanup-Crew]: https://github.com/rust-lang/rust/labels/ICEBreaker-Cleanup-Crew

The "Cleanup Crew" are focused on improving bug reports. Specifically,
the goal is to try to ensure that every bug report has all the
information that will be needed for someone to fix it:

* a minimal, standalone example that shows the problem
* links to duplicates or related bugs
* if the bug is a regression (something that used to work, but no longer does),
  then a bisection to the PR or nightly that caused the regression

This kind of cleanup is invaluable in getting bugs fixed. Better
still, it can be done by anybody who knows Rust, without any
particularly deep knowledge of the compiler.

Let's look a bit at the workflow for doing "cleanup crew" actions.

## Finding a minimal, standalone example

Here the ultimate goal is to produce an example that reproduces the same
problem but without relying on any external crates. Such a test ought to contain
as little code as possible, as well. This will make it much easier to isolate the problem.

However, even if the "ultimate minimal test" cannot be achieved, it's
still useful to post incremental minimizations. For example, if you
can eliminate some of the external dependencies, that is helpful, and
so forth. 

It's particularly useful to reduce to an example that works
in the [Rust playground](https://play.rust-lang.org/), rather than
requiring people to checkout a cargo build.

There are many resources for how to produce minimized test cases. Here
are a few:

* The [rust-reduce](https://github.com/jethrogb/rust-reduce) tool can try to reduce
  code automatically.
  * The [C-reduce](https://github.com/csmith-project/creduce) tool also works
    on Rust code, though it requires that you start from a single
    file. (A post explaining how to do it can be found [here](https://insaneinside.net/2017/09/12/whole-crate-bug-reduction-with-creduce.html).)
* pnkfelix's [Rust Bug Minimization Patterns] blog post
  * This post focuses on "heavy bore" techniques, where you are
    starting with a large, complex cargo project that you wish to
    narrow down to something standalone.

[Rust Bug Minimization Patterns]: http://blog.pnkfx.org/blog/2019/11/18/rust-bug-minimization-patterns/

## Links to duplicate or related bugs

If you are on the "Cleanup Crew", you will sometimes see multiple bug
reports that seem very similar. You can link one to the other just by
mentioning the other bug number in a Github comment. Sometimes it is
useful to close duplicate bugs. But if you do so, you should always
copy any test case from the bug you are closing to the other bug that
remains open, as sometimes duplicate-looking bugs will expose
different facets of the same problem.

## Bisecting regressions

For regressions (something that used to work, but no longer does), it
is super useful if we can figure out precisely when the code stopped
working.  The gold standard is to be able to identify the precise
**PR** that broke the code, so we can ping the author, but even
narrowing it down to a nightly build is helpful, especially as that
then gives us a range of PRs. (One other challenge is that we
sometimes land "rollup" PRs, which combine multiple PRs into one.)

### cargo-bisect-rustc

To help in figuring out the cause of a regression we have a tool
called [cargo-bisect-rustc]. It will automatically download and test
various builds of rustc. For recent regressions, it is even able to
use the builds from our CI to track down the regression to a specific
PR; for older regressions, it will simply identify a nightly.

To learn to use [cargo-bisect-rustc], check out [this blog post][learn], which
gives a quick introduction to how it works. Additionally, there is a [Guide]
which goes into more detail on how to use it. You can also ask questions at
the Zulip stream [`#t-compiler/cargo-bisect-rustc`][zcbr], or help in
improving the tool.

[cargo-bisect-rustc]: https://github.com/rust-lang/cargo-bisect-rustc/
[learn]: https://blog.rust-lang.org/inside-rust/2019/12/18/bisecting-rust-compiler.html
[zcbr]: https://rust-lang.zulipchat.com/#narrow/stream/217417-t-compiler.2Fcargo-bisect-rustc
[Guide]: https://rust-lang.github.io/cargo-bisect-rustc/
