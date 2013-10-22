## How to submit a bug report

If you're just reporting a bug, please see:

https://github.com/mozilla/rust/wiki/HOWTO-submit-a-Rust-bug-report

## Pull request procedure

Pull requests should be targeted at Rust's `master` branch.
Before pushing to your Github repo and issuing the pull request,
please do two things:

1. [Rebase](http://git-scm.com/book/en/Git-Branching-Rebasing) your
   local changes against the `master` branch. Resolve any conflicts
   that arise.

2. Run the full Rust test suite with the `make check` command.  You're
   not off the hook even if you just stick to documentation; code
   examples in the docs are tested as well!

Pull requests will be treated as "review requests", and we will give
feedback we expect to see corrected on
[style](https://github.com/mozilla/rust/wiki/Note-style-guide) and
substance before pulling.  Changes contributed via pull request should
focus on a single issue at a time, like any other.  We will not accept
pull-requests that try to "sneak" unrelated changes in.

Normally, all pull requests must include regression tests (see
[Note-testsuite](https://github.com/mozilla/rust/wiki/Note-testsuite))
that test your change.  Occasionally, a change will be very difficult
to test for.  In those cases, please include a note in your commit
message explaining why.

In the licensing header at the beginning of any files you change,
please make sure the listed date range includes the current year.  For
example, if it's 2013, and you change a Rust file that was created in
2010, it should begin:

```
// Copyright 2010-2013 The Rust Project Developers.
```

For more details, please refer to
[Note-development-policy](https://github.com/mozilla/rust/wiki/Note-development-policy).
