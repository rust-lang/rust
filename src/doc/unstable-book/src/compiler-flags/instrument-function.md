# `instrument-function`

Rust exposes several mechanisms to instrument functions which work with
existing platform tooling:

* xray
* mcount (prof/gprof style profiling)
* fentry (prof/gprof style profiling, though primarily used by the linux kernel).

These options are mutually exclusive. Only one can be enabled when compiling a
crate. These do not alter ABI, but may require implicit symbols to exist at
link time.

These are exposed via the `-Z instrument-function={mcount|fentry|xray|none}`
option.

A builtin attribute `instrument_fn` can be used to enable (as may be needed
for xray), or disable instrumentation of the function. E.g.:

```rust,no_run
#![feature(instrument_fn)]
#[instrument_fn = "off"]
fn non_instrumented_function() {
}
```

# `-Z instrument-function=mcount`

mcount instrumentation is a mechanism to insert a counting function call into
the prologue of functions. This is traditionally used to perform profiling of
a binary. In more recent years, mcount has been to implement features beyond
profiling, such as logging, function patching, and tracing (see linux's
[ftrace](https://docs.kernel.org/trace/ftrace.html)). This option is similiar
to the `-pg` option provided by clang and gcc. This option also requires
a frame-pointer to be enabled.

This option only applies to the crate being compiled, and

This feature is enabled via `-Zinstrument-function=mcount`, and an attribute
is provided to prevent instrumentation of functions.

On linux, an example might look like:
```rust,no_run
fn main() {
  println!("Hello world!");
}
```

And compiling and running for gprof a fedora 44 x86-64 host:

```shell
$ rustc main.rs foo.rs -Zinstrument-function=mcount -C link-args=/usr/lib64/gcrt1.o -C link-self-contained
$ ./main
$ gprof main gmon.out
```

gprof replaces parts of the C runtime implicitly linked into a binary. The above example is not
suitable for most cases as rustc does not yet know how to substitude crt\*.o for gcrt\*.o when
linking a binary.

# `-Z instrument-function=fentry`

On some targets, a more specialized form of mcount is available, named `fentry`. Unlike
mcount, a frame-pointer is not required, and this is guaranteed to be called at function
entry (hence the name).

This support is restricted to fewer targets. Today, only x86 and s390x support this, and
furthermore, only s390x supports advanced usage described below.

## Advanced usage

On supported targets, thee behavior of instrumentation can be further configured with the
`-Zinstrument-mcount-opts` flag. It supports the following options:

* `=record`: record the location of each call (or nop placeholder) into a section named
             `__mcount_loc`. This can be used to toggle the counting function at runtime.

* `=no-call`: insert nop's which can be replaced by a call to a counting function.

# `-Z instrument-function=xray`

The tracking issue for the xray feature is: [#102921](https://github.com/rust-lang/rust/issues/102921).

Enable generation of NOP sleds for XRay function tracing instrumentation.
For more information on XRay,
read [LLVM documentation](https://llvm.org/docs/XRay.html),
and/or the [XRay whitepaper](http://research.google.com/pubs/pub45287.html).

Set the `-Z instrument-function=xray` compiler flag in order to enable XRay instrumentation.

  - `-Z instrument-function=xray` – use the default settings
  - `-Z instrument-function=xray -Z instrument-xray-opts=skip-exit` – configure a custom setting
  - `-Z instrument-function=xray -Z instrument-xray-opts=ignore-loops,instruction-threshold=300` –
    multiple settings separated by commas

Supported options:

  - `always` – force instrumentation of all functions
  - `never` – do no instrument any functions
  - `ignore-loops` – ignore presence of loops,
    instrument functions based only on instruction count
  - `instruction-threshold=10` – set a different instruction threshold for instrumentation
  - `skip-entry` – do no instrument function entry
  - `skip-exit` – do no instrument function exit

The default settings are:

  - instrument both entry & exit from functions
  - instrument functions with at least 200 instructions,
    or containing a non-trivial loop

Note that `-Z instrument-function=xray` only enables generation of NOP sleds
which on their own don't do anything useful.
In order to actually trace the functions,
you will need to link a separate runtime library of your choice,
such as Clang's [XRay Runtime Library](https://www.llvm.org/docs/XRay.html#xray-runtime-library).
