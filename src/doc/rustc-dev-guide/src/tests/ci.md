# Testing with CI

## Testing infrastructure

<!-- date-check: may 2024 -->
When a Pull Request is opened on GitHub, [GitHub Actions] will automatically
launch a build that will run all tests on some configurations
(x86_64-gnu-llvm-X linux, x86_64-gnu-tools linux, mingw-check linux and mingw-check-tidy linux).
In essence, each runs `./x test` with various different options.

The integration bot [bors] is used for coordinating merges to the master branch.
When a PR is approved, it goes into a [queue] where merges are tested one at a
time on a wide set of platforms using GitHub Actions. Due to the limit on the
number of parallel jobs, we run the main CI jobs under the [rust-lang-ci] organization
(in contrast to PR CI jobs, which run under `rust-lang` directly).
Most platforms only run the build steps, some run a restricted set of tests,
only a subset run the full suite of tests (see Rust's [platform tiers]).

If everything passes, then all the distribution artifacts that were
generated during the CI run are published.

[GitHub Actions]: https://github.com/rust-lang/rust/actions
[rust-lang-ci]: https://github.com/rust-lang-ci/rust/actions
[bors]: https://github.com/rust-lang/homu
[queue]: https://bors.rust-lang.org/queue/rust
[platform tiers]: https://forge.rust-lang.org/release/platform-support.html#rust-platform-support

## Using CI to test

In some cases, a PR may run into problems with running tests on a particular
platform or configuration.
If you can't run those tests locally, don't hesitate to use CI resources to
try out a fix.

As mentioned above, opening or updating a PR will only run on a small subset
of configurations.
Only when a PR is approved will it go through the full set of test configurations.
However, you can try one of those configurations in your PR before it is approved.
For example, if a Windows build fails, but you don't have access to a Windows
machine, you can try running the Windows job that failed on CI within your PR
after pushing a possible fix.

To do this, you'll need to edit [`src/ci/github-actions/jobs.yml`]. It contains three
sections that affect which CI jobs will be executed:
- The `pr` section defines everything that will run after a push to a PR.
- The `try` section defines job(s) that are run when you ask for a try build using `@bors try`. 
- The `auto` section defines the full set of tests that are run after a PR is approved and before
it is merged into the main branch.

You can copy one of the definitions from the `auto` section to the `pr` or `try` sections.
For example, the `x86_64-msvc` job is responsible for running the 64-bit MSVC tests.
You can copy it to the `pr` section to cause it to be executed after a commit is pushed to your
PR, like this:

```yaml
pr:
  ...
  - image: x86_64-gnu-tools
    <<: *job-linux-16c
  # this item was copied from the `auto` section
  # vvvvvvvvvvvvvvvvvv
  - image: x86_64-msvc
    env:
      RUST_CONFIGURE_ARGS: --build=x86_64-pc-windows-msvc --enable-profiler
      SCRIPT: make ci-msvc
    <<: *job-windows-8c
```

Then, you can commit the file and push to GitHub. GitHub Actions should launch the tests.

After you have finished your tests, don't forget to remove any changes you have made to `jobs.yml`.

If you need to make more complex modifications to CI, you will need to modify
[`.github/workflows/ci.yml`] and possibly also
[`src/ci/github-actions/calculate-job-matrix.py`].

Although you are welcome to use CI, just be conscious that this is a shared
resource with limited concurrency.
Try not to enable too many jobs at once (one or two should be sufficient in
most cases).

[`src/ci/github-actions/jobs.yml`]: https://github.com/rust-lang/rust/blob/master/src/ci/github-actions/jobs.yml
[`.github/workflows/ci.yml`]: https://github.com/rust-lang/rust/blob/master/.github/workflows/ci.yml
[`src/ci/github-actions/calculate-job-matrix.py`]: https://github.com/rust-lang/rust/blob/master/src/ci/github-actions/calculate-job-matrix.py
