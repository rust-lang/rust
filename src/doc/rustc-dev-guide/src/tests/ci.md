# Testing with CI

The primary goal of our CI system is to ensure that the `master` branch of
`rust-lang/rust` is always in a valid state and passes our test suite.

From a high-level point of view, when you open a pull request at
`rust-lang/rust`, the following will happen:

- A small [subset](#pull-request-builds) of tests and checks are run after each
  push to the PR. This should help catching common errors.
- When the PR is approved, the [bors] bot enqueues the PR into a [merge queue].
- Once the PR gets to the front of the queue, bors will create a merge commit
  and run the [full test suite](#auto-builds) on it. The merge commit either
  contains only one specific PR or it can be a ["rollup"](#rollups) which
  combines multiple PRs together, to save CI costs.
- Once the whole test suite finishes, two things can happen. Either CI fails
  with an error that needs to be addressed by the developer, or CI succeeds and
  the merge commit is then pushed to the `master` branch.

If you want to modify what gets executed on CI, see [Modifying CI
jobs](#modifying-ci-jobs).

## CI workflow

<!-- date-check: Oct 2024 -->

Our CI is primarily executed on [GitHub Actions], with a single workflow defined
in [`.github/workflows/ci.yml`], which contains a bunch of steps that are
unified for all CI jobs that we execute. When a commit is pushed to a
corresponding branch or a PR, the workflow executes the
[`src/ci/citool`] crate, which dynamically generates the specific CI
jobs that should be executed. This script uses the [`jobs.yml`] file as an
input, which contains a declarative configuration of all our CI jobs.

> Almost all build steps shell out to separate scripts. This keeps the CI fairly
> platform independent (i.e., we are not overly reliant on GitHub Actions).
> GitHub Actions is only relied on for bootstrapping the CI process and for
> orchestrating the scripts that drive the process.

In essence, all CI jobs run `./x test`, `./x dist` or some other command with
different configurations, across various operating systems, targets and
platforms. There are two broad categories of jobs that are executed, `dist` and
non-`dist` jobs.

- Dist jobs build a full release of the compiler for a specific platform,
  including all the tools we ship through rustup; Those builds are then uploaded
  to the `rust-lang-ci2` S3 bucket and are available to be locally installed
  with the [rustup-toolchain-install-master] tool. The same builds are also used
  for actual releases: our release process basically consists of copying those
  artifacts from `rust-lang-ci2` to the production endpoint and signing them.
- Non-dist jobs run our full test suite on the platform, and the test suite of
  all the tools we ship through rustup; The amount of stuff we test depends on
  the platform (for example some tests are run only on Tier 1 platforms), and
  some quicker platforms are grouped together on the same builder to avoid
  wasting CI resources.

Based on an input event (usually a push to a branch), we execute one of three
kinds of builds (sets of jobs).

1. PR builds
2. Auto builds
3. Try builds

[rustup-toolchain-install-master]: https://github.com/kennytm/rustup-toolchain-install-master

### Pull Request builds

After each push to a pull request, a set of `pr` jobs are executed. Currently,
these execute the `x86_64-gnu-llvm-X`, `x86_64-gnu-tools`, `mingw-check-1`, `mingw-check-2`
and `mingw-check-tidy` jobs, all running on Linux. These execute a relatively short
(~40 minutes) and lightweight test suite that should catch common issues. More
specifically, they run a set of lints, they try to perform a cross-compile check
build to Windows mingw (without producing any artifacts) and they test the
compiler using a *system* version of LLVM. Unfortunately, it would take too many
resources to run the full test suite for each commit on every PR.

> **Note on doc comments**
>
> Note that PR CI as of Oct 2024 <!-- datecheck --> by default does not try to
> run `./x doc xxx`. This means that if you have any broken intradoc links that
> would lead to `./x doc xxx` failing, it will happen very late into the full
> merge queue CI pipeline.
>
> Thus, it is a good idea to run `./x doc xxx` locally for any doc comment
> changes to help catch these early.

PR jobs are defined in the `pr` section of [`jobs.yml`]. They run under the
`rust-lang/rust` repository, and their results can be observed directly on the
PR, in the "CI checks" section at the bottom of the PR page.

### Auto builds

Before a commit can be merged into the `master` branch, it needs to pass our
complete test suite. We call this an `auto` build. This build runs tens of CI
jobs that exercise various tests across operating systems and targets. The full
test suite is quite slow; it can take two hours or more until all the `auto` CI
jobs finish.

Most platforms only run the build steps, some run a restricted set of tests,
only a subset run the full suite of tests (see Rust's [platform tiers]).

Auto jobs are defined in the `auto` section of [`jobs.yml`]. They are executed
on the `auto` branch under the `rust-lang/rust` repository and
their results can be seen [here](https://github.com/rust-lang/rust/actions),
although usually you will be notified of the result by a comment made by bors on
the corresponding PR.

At any given time, at most a single `auto` build is being executed. Find out
more [here](#merging-prs-serially-with-bors).

[platform tiers]: https://forge.rust-lang.org/release/platform-support.html#rust-platform-support

### Try builds

Sometimes we want to run a subset of the test suite on CI for a given PR, or
build a set of compiler artifacts from that PR, without attempting to merge it.
We call this a "try build". A try build is started after a user with the proper
permissions posts a PR comment with the `@bors try` command.

There are several use-cases for try builds:

- Run a set of performance benchmarks using our [rustc-perf] benchmark suite.
  For this, a working compiler build is needed, which can be generated with a
  try build that runs the [dist-x86_64-linux] CI job, which builds an optimized
  version of the compiler on Linux (this job is currently executed by default
  when you start a try build). To create a try build and schedule it for a
  performance benchmark, you can use the `@bors try @rust-timer queue` command
  combination.
- Check the impact of the PR across the Rust ecosystem, using a [crater] run.
  Again, a working compiler build is needed for this, which can be produced by
  the [dist-x86_64-linux] CI job.
- Run a specific CI job (e.g. Windows tests) on a PR, to quickly test if it
  passes the test suite executed by that job.

By default, if you send a comment with `@bors try`, the jobs defined in the `try` section of
[`jobs.yml`] will be executed. We call this mode a "fast try build". Such a try build
will not execute any tests, and it will allow compilation warnings. It is useful when you want to
get an optimized toolchain as fast as possible, for a crater run or performance benchmarks,
even if it might not be working fully correctly.

If you want to run a custom CI job in a try build and make sure that it passes all tests and does
not produce any compilation warnings, you can select CI jobs to be executed by adding lines
containing `try-job: <job pattern>` to the PR description. All such specified jobs will be executed
in the try build once the `@bors try` command is used on the PR.

Each pattern can either be an exact name of a job or a glob pattern that matches multiple jobs,
for example `*msvc*` or `*-alt`. You can start at most 20 jobs in a single try build. When using
glob patterns, you might want to wrap them in backticks (`` ` ``) to avoid GitHub rendering
the pattern as Markdown.

> **Using `try-job` PR description directives**
>
> 1. Identify which set of try-jobs you would like to exercise. You can
>    find the name of the CI jobs in [`jobs.yml`].
>
> 2. Amend PR description to include a set of patterns (usually at the end
>    of the PR description), for example:
>
>    ```text
>    This PR fixes #123456.
>
>    try-job: x86_64-msvc
>    try-job: test-various
>    try-job: `*-alt`
>    ```
>
>    Each `try-job` pattern must be on its own line.
>
> 3. Run the prescribed try jobs with `@bors try`. As aforementioned, this
>    requires the user to either (1) have `try` permissions or (2) be delegated
>    with `try` permissions by `@bors delegate` by someone who has `try`
>    permissions.
>
> Note that this is usually easier to do than manually edit [`jobs.yml`].
> However, it can be less flexible because you cannot adjust the set of tests
> that are exercised this way.

Try jobs are defined in the `try` section of [`jobs.yml`]. They are executed on
the `try` branch under the `rust-lang/rust` repository and
their results can be seen [here](https://github.com/rust-lang/rust/actions),
although usually you will be notified of the result by a comment made by bors on
the corresponding PR.

Note that if you start the default try job using `@bors try`, it will skip building several `dist` components and running post-optimization tests, to make the build duration shorter. If you want to execute the full build as it would happen before a merge, add an explicit `try-job` pattern with the name of the default try job (currently `dist-x86_64-linux`).

Multiple try builds can execute concurrently across different PRs.

<div class="warning">

Bors identifies try jobs by commit hash. This means that if you have two PRs
containing the same (latest) commits, running `@bors try` will result in the
*same* try job and it really confuses `bors`. Please refrain from doing so.

</div>

[rustc-perf]: https://github.com/rust-lang/rustc-perf
[crater]: https://github.com/rust-lang/crater

### Modifying CI jobs

If you want to modify what gets executed on our CI, you can simply modify the
`pr`, `auto` or `try` sections of the [`jobs.yml`] file.

You can also modify what gets executed temporarily, for example to test a
particular platform or configuration that is challenging to test locally (for
example, if a Windows build fails, but you don't have access to a Windows
machine). Don't hesitate to use CI resources in such situations to try out a
fix!

You can perform an arbitrary CI job in two ways:
- Use the [try build](#try-builds) functionality, and specify the CI jobs that
  you want to be executed in try builds in your PR description.
- Modify the [`pr`](#pull-request-builds) section of `jobs.yml` to specify which
  CI jobs should be executed after each push to your PR. This might be faster
  than repeatedly starting try builds.

To modify the jobs executed after each push to a PR, you can simply copy one of
the job definitions from the `auto` section to the `pr` section. For example,
the `x86_64-msvc` job is responsible for running the 64-bit MSVC tests. You can
copy it to the `pr` section to cause it to be executed after a commit is pushed
to your PR, like this:

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

Then you can commit the file and push it to your PR branch on GitHub. GitHub
Actions should then execute this CI job after each push to your PR.

<div class="warning">

**After you have finished your experiments, don't forget to remove any changes
you have made to `jobs.yml`, if they were supposed to be temporary!**

A good practice is to prefix `[WIP]` in PR title while still running try jobs
and `[DO NOT MERGE]` in the commit that modifies the CI jobs for testing
purposes.
</div>

Although you are welcome to use CI, just be conscious that this is a shared
resource with limited concurrency. Try not to enable too many jobs at once (one
or two should be sufficient in most cases).

## Merging PRs serially with bors

CI services usually test the last commit of a branch merged with the last commit
in `master`, and while that’s great to check if the feature works in isolation,
it doesn’t provide any guarantee the code is going to work once it’s merged.
Breakages like these usually happen when another, incompatible PR is merged
after the build happened.

To ensure a `master` branch that works all the time, we forbid manual merges.
Instead, all PRs have to be approved through our bot, [bors] (the software
behind it is called [homu]). All the approved PRs are put in a [merge queue]
(sorted by priority and creation date) and are automatically tested one at the
time. If all the builders are green, the PR is merged, otherwise the failure is
recorded and the PR will have to be re-approved again.

Bors doesn’t interact with CI services directly, but it works by pushing the
merge commit it wants to test to specific branches (like `auto` or `try`), which
are configured to execute CI checks. Bors then detects the outcome of the build
by listening for either Commit Statuses or Check Runs. Since the merge commit is
based on the latest `master` and only one can be tested at the same time, when
the results are green, `master` is fast-forwarded to that merge commit.

Unfortunately testing a single PR at the time, combined with our long CI (~2
hours for a full run), means we can’t merge too many PRs in a single day, and a
single failure greatly impacts our throughput for the day. The maximum number of
PRs we can merge in a day is around ~10.

The large CI run times and requirement for a large builder pool is largely due
to the fact that full release artifacts are built in the `dist-` builders. This
is worth it because these release artifacts:

- Allow perf testing even at a later date.
- Allow bisection when bugs are discovered later.
- Ensure release quality since if we're always releasing, we can catch problems
  early.

### Rollups

Some PRs don’t need the full test suite to be executed: trivial changes like
typo fixes or README improvements *shouldn’t* break the build, and testing every
single one of them for 2+ hours is a big waste of time. To solve this, we
regularly create a "rollup", a PR where we merge several pending trivial PRs so
they can be tested together. Rollups are created manually by a team member using
the "create a rollup" button on the [merge queue]. The team member uses their
judgment to decide if a PR is risky or not, and are the best tool we have at the
moment to keep the queue in a manageable state.

## Docker

All CI jobs, except those on macOS and Windows, are executed inside that
platform’s custom [Docker container]. This has a lot of advantages for us:

- The build environment is consistent regardless of the changes of the
  underlying image (switching from the trusty image to xenial was painless for
  us).
- We can use ancient build environments to ensure maximum binary compatibility,
  for example [using older CentOS releases][dist-x86_64-linux] on our Linux
  builders.
- We can avoid reinstalling tools (like QEMU or the Android emulator) every time
  thanks to Docker image caching.
- Users can run the same tests in the same environment locally by just running
  `cargo run --manifest-path src/ci/citool/Cargo.toml run-local <job-name>`, which is awesome to debug failures. Note that there are only linux docker images available locally due to licensing and
  other restrictions.

The docker images prefixed with `dist-` are used for building artifacts while
those without that prefix run tests and checks.

We also run tests for less common architectures (mainly Tier 2 and Tier 3
platforms) in CI. Since those platforms are not x86 we either run everything
inside QEMU or just cross-compile if we don’t want to run the tests for that
platform.

These builders are running on a special pool of builders set up and maintained
for us by GitHub.

[Docker container]: https://github.com/rust-lang/rust/tree/master/src/ci/docker

## Caching

Our CI workflow uses various caching mechanisms, mainly for two things:

### Docker images caching

The Docker images we use to run most of the Linux-based builders take a *long*
time to fully build. To speed up the build, we cache them using [Docker registry
caching], with the intermediate artifacts being stored on [ghcr.io]. We also
push the built Docker images to ghcr, so that they can be reused by other tools
(rustup) or by developers running the Docker build locally (to speed up their
build).

Since we test multiple, diverged branches (`master`, `beta` and `stable`), we
can’t rely on a single cache for the images, otherwise builds on a branch would
override the cache for the others. Instead, we store the images under different
tags, identifying them with a custom hash made from the contents of all the
Dockerfiles and related scripts.

The CI calculates a hash key, so that the cache of a Docker image is
invalidated if one of the following changes:

- Dockerfile
- Files copied into the Docker image in the Dockerfile
- The architecture of the GitHub runner (x86 or ARM)

[ghcr.io]: https://github.com/rust-lang/rust/pkgs/container/rust-ci
[Docker registry caching]: https://docs.docker.com/build/cache/backends/registry/

### LLVM caching with sccache

We build some C/C++ stuff in various CI jobs, and we rely on [sccache] to cache
the intermediate LLVM artifacts. Sccache is a distributed ccache developed by
Mozilla, which can use an object storage bucket as the storage backend.

With sccache there's no need to calculate the hash key ourselves. Sccache
invalidates the cache automatically when it detects changes to relevant inputs,
such as the source code, the version of the compiler, and important environment
variables.
So we just pass the sccache wrapper on top of cargo and sccache does the rest.

We store the persistent artifacts on the S3 bucket `rust-lang-ci-sccache2`. So
when the CI runs, if sccache sees that LLVM is being compiled with the same C/C++
compiler and the LLVM source code is the same, sccache retrieves the individual
compiled translation units from S3.

[sccache]: https://github.com/mozilla/sccache

## Custom tooling around CI

During the years we developed some custom tooling to improve our CI experience.

### Rust Log Analyzer to show the error message in PRs

The build logs for `rust-lang/rust` are huge, and it’s not practical to find
what caused the build to fail by looking at the logs. To improve the developers’
experience we developed a bot called [Rust Log Analyzer][rla] (RLA) that
receives the build logs on failure and extracts the error message automatically,
posting it on the PR.

The bot is not hardcoded to look for error strings, but was trained with a bunch
of build failures to recognize which lines are common between builds and which
are not. While the generated snippets can be weird sometimes, the bot is pretty
good at identifying the relevant lines even if it’s an error we've never seen
before.

[rla]: https://github.com/rust-lang/rust-log-analyzer

### Toolstate to support allowed failures

The `rust-lang/rust` repo doesn’t only test the compiler on its CI, but also a
variety of tools and documentation. Some documentation is pulled in via git
submodules. If we blocked merging rustc PRs on the documentation being fixed, we
would be stuck in a chicken-and-egg problem, because the documentation's CI
would not pass since updating it would need the not-yet-merged version of rustc
to test against (and we usually require CI to be passing).

To avoid the problem, submodules are allowed to fail, and their status is
recorded in [rust-toolstate]. When a submodule breaks, a bot automatically pings
the maintainers so they know about the breakage, and it records the failure on
the toolstate repository. The release process will then ignore broken tools on
nightly, removing them from the shipped nightlies.

While tool failures are allowed most of the time, they’re automatically
forbidden a week before a release: we don’t care if tools are broken on nightly
but they must work on beta and stable, so they also need to work on nightly a
few days before we promote nightly to beta.

More information is available in the [toolstate documentation].

[rust-toolstate]: https://rust-lang-nursery.github.io/rust-toolstate
[toolstate documentation]: https://forge.rust-lang.org/infra/toolstate.html

## Public CI dashboard

To monitor the Rust CI, you can have a look at the [public dashboard] maintained by the infra-team.

These are some useful panels from the dashboard:

- Pipeline duration: check how long the auto builds takes to run.
- Top slowest jobs: check which jobs are taking the longest to run.
- Change in median job duration: check what jobs are slowest than before. Useful
  to detect regressions.
- Top failed jobs: check which jobs are failing the most.

To learn more about the dashboard, see the [Datadog CI docs].

[Datadog CI docs]: https://docs.datadoghq.com/continuous_integration/
[public dashboard]: https://p.datadoghq.com/sb/3a172e20-e9e1-11ed-80e3-da7ad0900002-b5f7bb7e08b664a06b08527da85f7e30

## Determining the CI configuration

If you want to determine which `bootstrap.toml` settings are used in CI for a
particular job, it is probably easiest to just look at the build log. To do
this:

1. Go to
   <https://github.com/rust-lang/rust/actions?query=branch%3Aauto+is%3Asuccess>
   to find the most recently successful build, and click on it.
2. Choose the job you are interested in on the left-hand side.
3. Click on the gear icon and choose "View raw logs"
4. Search for the string "Configure the build"
5. All of the build settings are listed below that starting with the
   `configure:` prefix.

[GitHub Actions]: https://github.com/rust-lang/rust/actions
[`jobs.yml`]: https://github.com/rust-lang/rust/blob/master/src/ci/github-actions/jobs.yml
[`.github/workflows/ci.yml`]: https://github.com/rust-lang/rust/blob/master/.github/workflows/ci.yml
[`src/ci/citool`]: https://github.com/rust-lang/rust/blob/master/src/ci/citool
[bors]: https://github.com/bors
[homu]: https://github.com/rust-lang/homu
[merge queue]: https://bors.rust-lang.org/queue/rust
[dist-x86_64-linux]: https://github.com/rust-lang/rust/blob/master/src/ci/docker/host-x86_64/dist-x86_64-linux/Dockerfile
