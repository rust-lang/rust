# Performance testing

## rustc-perf

A lot of work is put into improving the performance of the compiler and
preventing performance regressions.

The [rustc-perf](https://github.com/rust-lang/rustc-perf) project provides
several services for testing and tracking performance. It provides hosted
infrastructure for running benchmarks as a service. At this time, only
`x86_64-unknown-linux-gnu` builds are tracked.

A "perf run" is used to compare the performance of the compiler in different
configurations for a large collection of popular crates. Different
configurations include "fresh builds", builds with incremental compilation, etc.

The result of a perf run is a comparison between two versions of the compiler
(by their commit hashes).

You can also use `rustc-perf` to manually benchmark and profile the compiler
[locally](../profiling/with_rustc_perf.md).

### Automatic perf runs

After every PR is merged, a suite of benchmarks are run against the compiler.
The results are tracked over time on the <https://perf.rust-lang.org/> website.
Any changes are noted in a comment on the PR.

### Manual perf runs

Additionally, performance tests can be ran before a PR is merged on an as-needed
basis. You should request a perf run if your PR may affect performance,
especially if it can affect performance adversely.

To evaluate the performance impact of a PR, write this comment on the PR:

`@bors try @rust-timer queue`

> **Note**: Only users authorized to do perf runs are allowed to post this
> comment. Teams that are allowed to use it are tracked in the [Teams
> repository](https://github.com/rust-lang/team) with the `perf = true` value in
> the `[permissions]` section (and bors permissions are also required). If you
> are not on one of those teams, feel free to ask for someone to post it for you
> (either on Zulip or ask the assigned reviewer).

This will first tell bors to do a "try" build which do a full release build for
`x86_64-unknown-linux-gnu`. After the build finishes, it will place it in the
queue to run the performance suite against it. After the performance tests
finish, the bot will post a comment on the PR with a summary and a link to a
full report.

If you want to do a perf run for an already built artifact (e.g. for a previous
try build that wasn't benchmarked yet), you can run this instead:

`@rust-timer build <commit-sha>`

You cannot benchmark the same artifact twice though.

More information about the available perf bot commands can be found
[here](https://perf.rust-lang.org/help.html). 

More details about the benchmarking process itself are available in the [perf
collector
documentation](https://github.com/rust-lang/rustc-perf/blob/master/collector/README.md).
