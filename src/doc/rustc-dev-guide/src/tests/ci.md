# Testing with CI

## Testing infrastructure

<!-- date-check: oct 2022 -->
When a Pull Request is opened on GitHub, [GitHub Actions] will automatically
launch a build that will run all tests on some configurations
(x86_64-gnu-llvm-13 linux, x86_64-gnu-tools linux, and mingw-check linux).
In essence, each runs `./x test` with various different options.

The integration bot [bors] is used for coordinating merges to the master branch.
When a PR is approved, it goes into a [queue] where merges are tested one at a
time on a wide set of platforms using GitHub Actions. Due to the limit on the
number of parallel jobs, we run CI under the [rust-lang-ci] organization except
for PRs.
Most platforms only run the build steps, some run a restricted set of tests,
only a subset run the full suite of tests (see Rust's [platform tiers]).

If everything passes, then all of the distribution artifacts that were
generated during the CI run are published.

[GitHub Actions]: https://github.com/rust-lang/rust/actions
[rust-lang-ci]: https://github.com/rust-lang-ci/rust/actions
[bors]: https://github.com/servo/homu
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

To do this, you'll need to edit [`src/ci/github-actions/ci.yml`].
The `jobs` section defines the jobs that will run.
The `jobs.pr` section defines everything that will run in a push to a PR.
The `jobs.auto` section defines the full set of tests that are run after a PR is approved.
You can copy one of the definitions from the `auto` section up to the `pr` section.

For example, the `x86_64-msvc-1` and `x86_64-msvc-2` jobs are responsible for
running the 64-bit MSVC tests.
You can copy those up to the `jobs.pr.strategy.matrix.include` section with
the other jobs.

The comment at the top of `ci.yml` will tell you to run this command:

```sh
./x run src/tools/expand-yaml-anchors
```

This will generate the true [`.github/workflows/ci.yml`] which is what GitHub
Actions uses.

Then, you can commit those two files and push to GitHub.
GitHub Actions should launch the tests.

After you have finished, don't forget to remove any changes you have made to `ci.yml`.

Although you are welcome to use CI, just be conscientious that this is a shared
resource with limited concurrency.
Try not to enable too many jobs at once (one or two should be sufficient in
most cases).

[`src/ci/github-actions/ci.yml`]: https://github.com/rust-lang/rust/blob/master/src/ci/github-actions/ci.yml
[`.github/workflows/ci.yml`]: https://github.com/rust-lang/rust/blob/master/.github/workflows/ci.yml#L1
