- Feature Name: Windows Subsystem
- Start Date: 2016-07-03
- RFC PR: [rust-lang/rfcs#1665](https://github.com/rust-lang/rfcs/pull/1665)
- Rust Issue: [rust-lang/rust#37499](https://github.com/rust-lang/rust/issues/37499)

# Summary
[summary]: #summary

Rust programs compiled for Windows will always allocate a console window on
startup. This behavior is controlled via the `SUBSYSTEM` parameter passed to the
linker, and so *can* be overridden with specific compiler flags. However, doing
so will bypass the Rust-specific initialization code in `libstd`, as when using
the MSVC toolchain, the entry point must be named `WinMain`.

This RFC proposes supporting this case explicitly, allowing `libstd` to
continue to be initialized correctly.

# Motivation
[motivation]: #motivation

The `WINDOWS` subsystem is commonly used on Windows: desktop applications
typically do not want to flash up a console window on startup.

Currently, using the `WINDOWS` subsystem from Rust is undocumented, and the
process is non-trivial when targeting the MSVC toolchain. There are a couple of
approaches, each with their own downsides:

## Define a WinMain symbol

A new symbol `pub extern "system" WinMain(...)` with specific argument
and return types must be declared, which will become the new entry point for
the program.

This is unsafe, and will skip the initialization code in `libstd`.

The GNU toolchain will accept either entry point.

## Override the entry point via linker options

This uses the same method as will be described in this RFC. However, it will
result in build scripts also being compiled for the `WINDOWS` subsystem, which
can cause additional console windows to pop up during compilation, making the
system unusable while a build is in progress.

# Detailed design
[design]: #detailed-design

When an executable is linked while compiling for a Windows target, it will be
linked for a specific *subsystem*. The subsystem determines how the operating
system will run the executable, and will affect the execution environment of
the program.

In practice, only two subsystems are very commonly used: `CONSOLE` and
`WINDOWS`, and from a user's perspective, they determine whether a console will
be automatically created when the program is started.

## New crate attribute

This RFC proposes two changes to solve this problem. The first is adding a
top-level crate attribute to allow specifying which subsystem to use:

`#![windows_subsystem = "windows"]`

Initially, the set of possible values will be `{windows, console}`, but may be
extended in future if desired.

The use of this attribute in a non-executable crate will result in a compiler
warning. If compiling for a non-Windows target, the attribute will be silently
ignored.

## Additional linker argument

For the GNU toolchain, this will be sufficient. However, for the MSVC toolchain,
the linker will be expecting a `WinMain` symbol, which will not exist.

There is some complexity to the way in which a different entry point is expected
when using the `WINDOWS` subsystem. Firstly, the C-runtime library exports two
symbols designed to be used as an entry point:
```
mainCRTStartup
WinMainCRTStartup
```

`LINK.exe` will use the subsystem to determine which of these symbols to use
as the default entry point if not overridden.

Each one performs some unspecified initialization of the CRT, before calling out
to a symbol defined within the program (`main` or `WinMain` respectively).

The second part of the solution is to pass an additional linker option when
targeting the MSVC toolchain:
`/ENTRY:mainCRTStartup`

This will override the entry point to always be `mainCRTStartup`. For
console-subsystem programs this will have no effect, since it was already the
default, but for `WINDOWS` subsystem programs, it will eliminate the need for
a `WinMain` symbol to be defined.

This command line option will always be passed to the linker, regardless of the
presence or absence of the `windows_subsystem` crate attribute, except when
the user specifies their own entry point in the linker arguments. This will
require `rustc` to perform some basic parsing of the linker options.

# Drawbacks
[drawbacks]: #drawbacks

- A new platform-specific crate attribute.
- The difficulty of manually calling the Rust initialization code is potentially
  a more general problem, and this only solves a specific (if common) case.
- The subsystem must be specified earlier than is strictly required: when
  compiling C/C++ code only the linker, not the compiler, needs to actually be
  aware of the subsystem.
- It is assumed that the initialization performed by the two CRT entry points
  is identical. This seems to currently be the case, and is unlikely to change
  as this technique appears to be used fairly widely.

# Alternatives
[alternatives]: #alternatives

- Only emit one of either `WinMain` or `main` from `rustc` based on a new
  command line option.

  This command line option would only be applicable when compiling an
  executable, and only for Windows platforms. No other supported platforms
  require a different entry point or additional linker arguments for programs
  designed to run with a graphical user interface.

  `rustc` will react to this command line option by changing the exported
  name of the entry point to `WinMain`, and passing additional arguments to
  the linker to configure the correct subsystem. A mismatch here would result
  in linker errors.

  A similar option would need to be added to `Cargo.toml` to make usage as
  simple as possible.

  There's some bike-shedding which can be done on the exact command line
  interface, but one possible option is shown below.

  Rustc usage:
  `rustc foo.rs --crate-subsystem windows`

  Cargo.toml
  ```toml
  [package]
  # ...

  [[bin]]
  name = "foo"
  path = "src/foo.rs"
  subsystem = "windows"
  ```

  The `crate-subsystem` command line option would exist on all platforms,
  but would be ignored when compiling for a non-Windows target, so as to
  support cross-compiling. If not compiling a binary crate, specifying the
  option is an error regardless of the target.

# Unresolved questions
[unresolved]: #unresolved-questions

None
