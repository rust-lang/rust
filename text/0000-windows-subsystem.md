- Feature Name: Windows Subsystem
- Start Date: 2016-07-03
- RFC PR: ____
- Rust Issue: ____

# Summary
[summary]: #summary

Rust programs compiled for windows will always flash up a console window on
startup. This behavior is controlled via the `SUBSYSTEM` parameter passed to the
linker, and so *can* be overridden with specific compiler flags. However, doing
so will bypass the rust-specific initialization code in `libstd`, as when using
the MSVC toolchain, the entry point must be named `WinMain`.

This RFC proposes supporting this case explicitly, allowing `libstd` to
continue to be initialized correctly.

# Motivation
[motivation]: #motivation

The `WINDOWS` subsystem is commonly used on windows: desktop applications
typically do not want to flash up a console window on startup.

Currently, using the `WINDOWS` subsystem from rust is undocumented, and the
process is non-trivial when targeting the MSVC toolchain:

A new symbol `pub extern "system" WinMain(...)` with specific argument
and return types must be declared, which will become the new entry point for
the program.

This is unsafe, and will skip the initialization code in `libstd`.

The GNU toolchain will accept either entry point.

# Detailed design
[design]: #detailed-design

When an executable is linked while compiling for a windows target, it will be
linked for a specific *subsystem*. The subsystem determines how the operating
system will run the executable, and will affect the execution environment of
the program.

In practice, only two subsystems are very commonly used: `CONSOLE` and
`WINDOWS`, and from a user's perspective, they determine whether a console will
be automatically created when the program is started.

The solution this RFC proposes is to always export both `main` and `WinMain`
symbols from rust executables compiled for windows. The `WinMain` function
will simply delegate to the `main` function.

The exact signature is:
```
pub extern "system" WinMain(
    hInstance: HINSTANCE,
    hPrevInstance: HINSTANCE,
    lpCmdLine: LPSTR,
    nCmdShow: i32
) -> i32;
```

Where `HINSTANCE` is a pointer-sized opaque handle, and `LPSTR` is a C-style
null terminated string.

All four parameters are either irrelevant or can be obtained easily through
other means:
- `hInstance` - Can be obtained via `GetModuleHandle`.
- `hPrevInstance` - Is always NULL.
- `lpCmdLine` - `libstd` already provides a function to get command line
  arguments.
- `nCmdShow` - Can be obtained via `GetStartupInfo`, although it's not actually
  needed any more (the OS will automatically hide/show the first window created).

The end result is that rust programs will "just work" when the subsystem is
overridden via custom linker arguments, and does not require `rustc` to
parse those linker arguments.

A possible future extension would be to add additional command-line options to
`rustc` (and in turn, `Cargo.toml`) to specify the subsystem directly. `rustc`
would automatically translate this into the correct linker arguments for
whichever linker is actually being used.

# Drawbacks
[drawbacks]: #drawbacks

- Additional platform-specific code.
- The difficulty of manually calling the rust initialization code is potentially
  a more general problem, and this only solves a specific (if common) case.
- This is a breaking change for any crates which already export a `WinMain`
  symbol. It is likely that only executable crates would export this symbol,
  so the knock-on effect on crate dependencies should be non-existent.

  A possible work-around for this is described below.

# Alternatives
[alternatives]: #alternatives

- Only emit one of either `WinMain` or `main` from `rustc` based on a new
  command line option.

  This command line option would only be applicable when compiling an
  executable, and only for windows platforms. No other supported platforms
  require a different entry point or additional linker arguments for programs
  designed to run with a graphical user interface.

  `rustc` will react to this command line option by changing the exported
  name of the entry point to `WinMain`, and passing additional arguments to
  the linker to configure the correct subsystem. A mismatch here would result
  in linker errors.

  A similar option would need to be added to `Cargo.toml` to make usage as
  simple as possible.

  There's some bike-shedding which can be done one the exact command line
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
  but would be ignored when compiling for a non-windows target, so as to
  support cross-compiling. If not compiling a binary crate, specifying the
  option is an error regardless of the target.

- Have `rustc` override the entry point when calling `link.exe`, and tell it to
  use `mainCRTStartup` instead of `winMainCRTStartup`. These are the "true"
  entry points of windows programs, which first initialize the C runtime
  library, and then call `main` or `WinMain` respectively.

  This is the simplest solution, and it will not have any serious backwards
  compatibility problems, since rust programs are already required to have a
  `main` function, even if `WinMain` has been separately defined. However, it
  relies on the two CRT functions to be interchangeable, although this does
  *appear* to be the case currently.

- Export both entry points as described in this RFC, but also add a `subsystem`
  function to `libstd` determine which subsystem was used at runtime.

  The `WinMain` function would first set an internal flag, and only then
  delegate to the `main` function.

  A function would be added to `std::os::windows`:

  `fn subsystem() -> &'static str`

  This would check the value of the internal flag, and return either `WINDOWS` or
  `CONSOLE` depending on which entry point was actually used.

  The `subsystem` function could be used to eg. redirect logging to a file if
  the program is being run on the `WINDOWS` subsystem. However, it would return
  an incorrect value if the initialization was skipped, such as if used as a
  library from an executable written in another language.

- Export both entry points as described in this RFC, but use the undocumented
  MSVC equivalent to weak symbols to avoid breaking existing code.

  The parameter `/alternatename:_WinMain@16=_RustWinMain@16` can be used to
  export `WinMain` only if it is not also exported elsewhere. This is completely
  undocumented, but is mentioned here: (http://stackoverflow.com/a/11529277).

# Unresolved questions
[unresolved]: #unresolved-questions

None
