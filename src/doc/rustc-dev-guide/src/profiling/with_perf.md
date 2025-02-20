# Profiling with perf

This is a guide for how to profile rustc with [perf](https://perf.wiki.kernel.org/index.php/Main_Page).

## Initial steps

- Get a clean checkout of rust-lang/master, or whatever it is you want
  to profile.
- Set the following settings in your `bootstrap.toml`:
  - `debuginfo-level = 1` - enables line debuginfo
  - `jemalloc = false` - lets you do memory use profiling with valgrind
  - leave everything else the defaults
- Run `./x build` to get a full build
- Make a rustup toolchain pointing to that result
  - see [the "build and run" section for instructions][b-a-r]

[b-a-r]: ../building/how-to-build-and-run.html#toolchain

## Gathering a perf profile

perf is an excellent tool on linux that can be used to gather and
analyze all kinds of information. Mostly it is used to figure out
where a program spends its time. It can also be used for other sorts
of events, though, like cache misses and so forth.

### The basics

The basic `perf` command is this:

```bash
perf record -F99 --call-graph dwarf XXX
```

The `-F99` tells perf to sample at 99 Hz, which avoids generating too
much data for longer runs (why 99 Hz you ask? It is often chosen
because it is unlikely to be in lockstep with other periodic
activity). The `--call-graph dwarf` tells perf to get call-graph
information from debuginfo, which is accurate. The `XXX` is the
command you want to profile. So, for example, you might do:

```bash
perf record -F99 --call-graph dwarf cargo +<toolchain> rustc
```

to run `cargo` -- here `<toolchain>` should be the name of the toolchain
you made in the beginning. But there are some things to be aware of:

- You probably don't want to profile the time spend building
  dependencies. So something like `cargo build; cargo clean -p $C` may
  be helpful (where `$C` is the crate name)
    - Though usually I just do `touch src/lib.rs` and rebuild instead. =)
- You probably don't want incremental messing about with your
  profile. So something like `CARGO_INCREMENTAL=0` can be helpful.

### Gathering a perf profile from a `perf.rust-lang.org` test

Often we want to analyze a specific test from `perf.rust-lang.org`.
The easiest way to do that is to use the [rustc-perf][rustc-perf]
benchmarking suite, this approach is described [here](with_rustc_perf.md).

Instead of using the benchmark suite CLI, you can also profile the benchmarks manually. First,
you need to clone the [rustc-perf][rustc-perf] repository:

```bash
$ git clone https://github.com/rust-lang/rustc-perf
```

and then find the source code of the test that you want to profile. Sources for the tests
are found in [the `collector/compile-benchmarks` directory][compile-time dir]
and [the `collector/runtime-benchmarks` directory][runtime dir]. So let's
go into the directory of a specific test; we'll use `clap-rs` as an example:

[rustc-perf]: https://github.com/rust-lang/rustc-perf
[compile-time dir]: https://github.com/rust-lang/rustc-perf/tree/master/collector/compile-benchmarks
[runtime dir]: https://github.com/rust-lang/rustc-perf/tree/master/collector/runtime-benchmarks

```bash
cd collector/compile-benchmarks/clap-3.1.6
```

In this case, let's say we want to profile the `cargo check`
performance. In that case, I would first run some basic commands to
build the dependencies:

```bash
# Setup: first clean out any old results and build the dependencies:
cargo +<toolchain> clean
CARGO_INCREMENTAL=0 cargo +<toolchain> check
```

(Again, `<toolchain>` should be replaced with the name of the
toolchain we made in the first step.)

Next: we want record the execution time for *just* the clap-rs crate,
running cargo check. I tend to use `cargo rustc` for this, since it
also allows me to add explicit flags, which we'll do later on.

```bash
touch src/lib.rs
CARGO_INCREMENTAL=0 perf record -F99 --call-graph dwarf cargo rustc --profile check --lib
```

Note that final command: it's a doozy! It uses the `cargo rustc`
command, which executes rustc with (potentially) additional options;
the `--profile check` and `--lib` options specify that we are doing a
`cargo check` execution, and that this is a library (not a binary).

At this point, we can use `perf` tooling to analyze the results. For example:

```bash
perf report
```

will open up an interactive TUI program. In simple cases, that can be
helpful. For more detailed examination, the [`perf-focus` tool][pf]
can be helpful; it is covered below.

**A note of caution.** Each of the rustc-perf tests is its own special
  snowflake. In particular, some of them are not libraries, in which
  case you would want to do `touch src/main.rs` and avoid passing
  `--lib`. I'm not sure how best to tell which test is which to be
  honest.

### Gathering NLL data

If you want to profile an NLL run, you can just pass extra options to
the `cargo rustc` command, like so:

```bash
touch src/lib.rs
CARGO_INCREMENTAL=0 perf record -F99 --call-graph dwarf cargo rustc --profile check --lib -- -Z borrowck=mir
```

[pf]: https://github.com/nikomatsakis/perf-focus

## Analyzing a perf profile with `perf focus`

Once you've gathered a perf profile, we want to get some information
about it. For this, I personally use [perf focus][pf]. It's a kind of
simple but useful tool that lets you answer queries like:

- "how much time was spent in function F" (no matter where it was called from)
- "how much time was spent in function F when it was called from G"
- "how much time was spent in function F *excluding* time spent in G"
- "what functions does F call and how much time does it spend in them"

To understand how it works, you have to know just a bit about
perf. Basically, perf works by *sampling* your process on a regular
basis (or whenever some event occurs). For each sample, perf gathers a
backtrace. `perf focus` lets you write a regular expression that tests
which functions appear in that backtrace, and then tells you which
percentage of samples had a backtrace that met the regular
expression. It's probably easiest to explain by walking through how I
would analyze NLL performance.

### Installing `perf-focus`

You can install perf-focus using `cargo install`:

```bash
cargo install perf-focus
```

### Example: How much time is spent in MIR borrowck?

Let's say we've gathered the NLL data for a test. We'd like to know
how much time it is spending in the MIR borrow-checker. The "main"
function of the MIR borrowck is called `do_mir_borrowck`, so we can do
this command:

```bash
$ perf focus '{do_mir_borrowck}'
Matcher    : {do_mir_borrowck}
Matches    : 228
Not Matches: 542
Percentage : 29%
```

The `'{do_mir_borrowck}'` argument is called the **matcher**. It
specifies the test to be applied on the backtrace. In this case, the
`{X}` indicates that there must be *some* function on the backtrace
that meets the regular expression `X`. In this case, that regex is
just the name of the function we want (in fact, it's a subset of the name;
the full name includes a bunch of other stuff, like the module
path). In this mode, perf-focus just prints out the percentage of
samples where `do_mir_borrowck` was on the stack: in this case, 29%.

**A note about c++filt.** To get the data from `perf`, `perf focus`
  currently executes `perf script` (perhaps there is a better
  way...). I've sometimes found that `perf script` outputs C++ mangled
  names. This is annoying. You can tell by running `perf script |
  head` yourself — if you see names like `5rustc6middle` instead of
  `rustc::middle`, then you have the same problem. You can solve this
  by doing:

```bash
perf script | c++filt | perf focus --from-stdin ...
```

This will pipe the output from `perf script` through `c++filt` and
should mostly convert those names into a more friendly format. The
`--from-stdin` flag to `perf focus` tells it to get its data from
stdin, rather than executing `perf focus`. We should make this more
convenient (at worst, maybe add a `c++filt` option to `perf focus`, or
just always use it — it's pretty harmless).

### Example: How much time does MIR borrowck spend solving traits?

Perhaps we'd like to know how much time MIR borrowck spends in the
trait checker. We can ask this using a more complex regex:

```bash
$ perf focus '{do_mir_borrowck}..{^rustc::traits}'
Matcher    : {do_mir_borrowck},..{^rustc::traits}
Matches    : 12
Not Matches: 1311
Percentage : 0%
```

Here we used the `..` operator to ask "how often do we have
`do_mir_borrowck` on the stack and then, later, some function whose
name begins with `rustc::traits`?" (basically, code in that module). It
turns out the answer is "almost never" — only 12 samples fit that
description (if you ever see *no* samples, that often indicates your
query is messed up).

If you're curious, you can find out exactly which samples by using the
`--print-match` option. This will print out the full backtrace for
each sample. The `|` at the front of the line indicates the part that
the regular expression matched.

### Example: Where does MIR borrowck spend its time?

Often we want to do more "explorational" queries. Like, we know that
MIR borrowck is 29% of the time, but where does that time get spent?
For that, the `--tree-callees` option is often the best tool. You
usually also want to give `--tree-min-percent` or
`--tree-max-depth`. The result looks like this:

```bash
$ perf focus '{do_mir_borrowck}' --tree-callees --tree-min-percent 3
Matcher    : {do_mir_borrowck}
Matches    : 577
Not Matches: 746
Percentage : 43%

Tree
| matched `{do_mir_borrowck}` (43% total, 0% self)
: | rustc_borrowck::nll::compute_regions (20% total, 0% self)
: : | rustc_borrowck::nll::type_check::type_check_internal (13% total, 0% self)
: : : | core::ops::function::FnOnce::call_once (5% total, 0% self)
: : : : | rustc_borrowck::nll::type_check::liveness::generate (5% total, 3% self)
: : : | <rustc_borrowck::nll::type_check::TypeVerifier<'a, 'b, 'tcx> as rustc::mir::visit::Visitor<'tcx>>::visit_mir (3% total, 0% self)
: | rustc::mir::visit::Visitor::visit_mir (8% total, 6% self)
: | <rustc_borrowck::MirBorrowckCtxt<'cx, 'tcx> as rustc_mir_dataflow::DataflowResultsConsumer<'cx, 'tcx>>::visit_statement_entry (5% total, 0% self)
: | rustc_mir_dataflow::do_dataflow (3% total, 0% self)
```

What happens with `--tree-callees` is that

- we find each sample matching the regular expression
- we look at the code that occurs *after* the regex match and try
  to build up a call tree

The `--tree-min-percent 3` option says "only show me things that take
more than 3% of the time". Without this, the tree often gets really
noisy and includes random stuff like the innards of
malloc. `--tree-max-depth` can be useful too, it just limits how many
levels we print.

For each line, we display the percent of time in that function
altogether ("total") and the percent of time spent in **just that
function and not some callee of that function** (self). Usually
"total" is the more interesting number, but not always.

### Relative percentages

By default, all in perf-focus are relative to the **total program
execution**. This is useful to help you keep perspective — often as
we drill down to find hot spots, we can lose sight of the fact that,
in terms of overall program execution, this "hot spot" is actually not
important. It also ensures that percentages between different queries
are easily compared against one another.

That said, sometimes it's useful to get relative percentages, so `perf
focus` offers a `--relative` option. In this case, the percentages are
listed only for samples that match (vs all samples). So for example we
could get our percentages relative to the borrowck itself
like so:

```bash
$ perf focus '{do_mir_borrowck}' --tree-callees --relative --tree-max-depth 1 --tree-min-percent 5
Matcher    : {do_mir_borrowck}
Matches    : 577
Not Matches: 746
Percentage : 100%

Tree
| matched `{do_mir_borrowck}` (100% total, 0% self)
: | rustc_borrowck::nll::compute_regions (47% total, 0% self) [...]
: | rustc::mir::visit::Visitor::visit_mir (19% total, 15% self) [...]
: | <rustc_borrowck::MirBorrowckCtxt<'cx, 'tcx> as rustc_mir_dataflow::DataflowResultsConsumer<'cx, 'tcx>>::visit_statement_entry (13% total, 0% self) [...]
: | rustc_mir_dataflow::do_dataflow (8% total, 1% self) [...]
```

Here you see that `compute_regions` came up as "47% total" — that
means that 47% of `do_mir_borrowck` is spent in that function. Before,
we saw 20% — that's because `do_mir_borrowck` itself is only 43% of
the total time (and `.47 * .43 = .20`).
