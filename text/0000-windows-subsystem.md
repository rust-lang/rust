- Feature Name: Windows Subsystem
- Start Date: 2016-07-03
- RFC PR: ____
- Rust Issue: ____

# Summary
[summary]: #summary

Rust programs compiled for windows will always flash up a console window on
startup. This behavior is controlled via the `SUBSYSTEM` parameter passed to the
linker, and so *can* be overridden with specific compiler flags. However, doing
so will bypass the rust-specific initialization code in `libstd`.

This RFC proposes supporting this case explicitly, allowing `libstd` to
continue to be initialized correctly.

# Motivation
[motivation]: #motivation

The `WINDOWS` subsystem is commonly used on windows: desktop applications
typically do not want to flash up a console window on startup.

Currently, using the `WINDOWS` subsystem from rust is undocumented, and the
process is non-trivial:

A new symbol `pub extern "system" WinMain(...)` with specific argument
and return types must be declared, which will become the new entry point for
the program.

This is unsafe, and will skip the initialization code in `libstd`.

# Detailed design
[design]: #detailed-design

When an executable is linked while compiling for a windows target, it will be
linked for a specific *Subsystem*. The subsystem determines how the operating
system will run the executable, and will affect the execution environment of
the program.

In practice, only two subsystems are very commonly used: `CONSOLE` and
`WINDOWS`, and from a user's perspective, they determine whether a console will
be automatically created when the program is started.

The solution this RFC proposes is to always export both `main` and `WinMain`
symbols from rust executables compiled for windows. The `WinMain` function
will simply delegate to the `main` function.

The end result is that rust programs will "just work" when the subsystem is
overridden via custom linker arguments, and does not require `rustc` to
parse those linker arguments.

A possible future extension would be to add additional command-line options to
`rustc` (and in turn, Cargo.toml) to specify the subsystem directly. `rustc`
would automatically translate this into the correct linker arguments for
whichever linker is actually being used.

# Drawbacks
[drawbacks]: #drawbacks

- Additional platform-specific code.
- The difficulty of manually calling the rust initialization code is potentially
  a more general problem, and this only solves a specific (if common) case.

# Alternatives
[alternatives]: #alternatives

- Choosing to emit only `WinMain` or `main` using `cfg` attributes or similar.

  This is problematic because it requires the compiler to know which subsystem
  is being compiled for, and all dependencies would have to be compiled for that
  specific subsystem.

  Pushing the decision further down the line, making it purely a
  link-time/run-time decision reduces the potential for binary incompatibility.

- Add a `subsystem` function to determine which subsystem was used.

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

# Unresolved questions
[unresolved]: #unresolved-questions

None
