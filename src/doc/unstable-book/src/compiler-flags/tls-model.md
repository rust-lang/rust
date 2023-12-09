# `tls_model`

The tracking issue for this feature is: None.

------------------------

Option `-Z tls-model` controls [TLS model](https://www.akkadia.org/drepper/tls.pdf) used to
generate code for accessing `#[thread_local]` `static` items.

Supported values for this option are:

- `global-dynamic` - General Dynamic TLS Model (alternatively called Global Dynamic) is the most
general option usable in all circumstances, even if the TLS data is defined in a shared library
loaded at runtime and is accessed from code outside of that library.
This is the default for most targets.
- `local-dynamic` - model usable if the TLS data is only accessed from the shared library or
executable it is defined in. The TLS data may be in a library loaded after startup (via `dlopen`).
- `initial-exec` - model usable if the TLS data is defined in the executable or in a shared library
loaded at program startup.
The TLS data must not be in a library loaded after startup (via `dlopen`).
- `local-exec` - model usable only if the TLS data is defined directly in the executable,
but not in a shared library, and is accessed only from that executable.
- `emulated` - Uses thread-specific data keys to implement emulated TLS.
It is like using a general-dynamic TLS model for all modes.

`rustc` and LLVM may use a more optimized model than specified if they know that we are producing
an executable rather than a library, or that the `static` item is private enough.
