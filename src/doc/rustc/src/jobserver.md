# Jobserver

Internally, `rustc` may take advantage of parallelism. `rustc` will coordinate
with the build system calling it if a [GNU Make jobserver] is passed in the
`MAKEFLAGS` environment variable. Other flags may have an effect as well, such
as [`CARGO_MAKEFLAGS`]. If a jobserver is not passed, then `rustc` will choose
the number of jobs to use.

Starting with Rust 1.76.0, `rustc` will warn if a jobserver appears to be
available but is not accessible, e.g.:

```console
$ echo 'fn main() {}' | MAKEFLAGS=--jobserver-auth=3,4 rustc -
warning: failed to connect to jobserver from environment variable `MAKEFLAGS="--jobserver-auth=3,4"`: cannot open file descriptor 3 from the jobserver environment variable value: Bad file descriptor (os error 9)
  |
  = note: the build environment is likely misconfigured
```

## Integration with build systems

The following subsections contain recommendations on how to integrate `rustc`
with build systems so that the jobserver is handled appropriately.

### GNU Make

When calling `rustc` from GNU Make, it is recommended that all `rustc`
invocations are marked as recursive in the `Makefile` (by prefixing the command
line with the `+` indicator), so that GNU Make enables the jobserver for them.
For instance:

<!-- ignore-tidy-tab -->

```make
x:
	+@echo 'fn main() {}' | rustc -
```

In particular, GNU Make 4.3 (a widely used version as of 2024) passes a simple
pipe jobserver in `MAKEFLAGS` even when it was not made available for the child
process, which in turn means `rustc` will warn about it. For instance, if the
`+` indicator is removed from the example above and GNU Make is called with e.g.
`make -j2`, then the aforementioned warning will trigger.

For calls to `rustc` inside `$(shell ...)` inside a recursive Make, one can
disable the jobserver manually by clearing the `MAKEFLAGS` variable, e.g.:

```make
S := $(shell MAKEFLAGS= rustc --print sysroot)

x:
	@$(MAKE) y

y:
	@echo $(S)
```

### CMake

CMake 3.28 supports the `JOB_SERVER_AWARE` option in its [`add_custom_target`]
command, e.g.:

```cmake
cmake_minimum_required(VERSION 3.28)
project(x)
add_custom_target(x
    JOB_SERVER_AWARE TRUE
    COMMAND echo 'fn main() {}' | rustc -
)
```

For earlier versions, when using CMake with the Makefile generator, one
workaround is to have [`$(MAKE)`] somewhere in the command so that GNU Make
treats it as a recursive Make call, e.g.:

```cmake
cmake_minimum_required(VERSION 3.22)
project(x)
add_custom_target(x
    COMMAND DUMMY_VARIABLE=$(MAKE) echo 'fn main() {}' | rustc -
)
```

[GNU Make jobserver]: https://www.gnu.org/software/make/manual/html_node/POSIX-Jobserver.html
[`CARGO_MAKEFLAGS`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html
[`add_custom_target`]: https://cmake.org/cmake/help/latest/command/add_custom_target.html
[`$(MAKE)`]: https://www.gnu.org/software/make/manual/html_node/MAKE-Variable.html
