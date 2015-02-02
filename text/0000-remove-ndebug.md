- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Remove official support for the `ndebug` config variable, replace the current usage of it with a
more appropriate `debug_assertions` compiler-provided config variable.

# Motivation

The usage of 'ndebug' to indicate a release build is a strange holdover from C/C++. It is not used
much and is easy to forget about. Since it used like any other value passed to the `cfg` flag, it
does not interact with other flags such as `-g` or `-O`.

The only current users of `ndebug` are the implementations of the `debug_assert!` macro. At the
time of this writing integer overflow checking is will also be controlled by this variable. Since
the optimisation setting does not influence `ndebug`, this means that code that the user expects to
be optimised will still contain the overflow checking logic. Similarly, `debug_assert!` invocations
are not removed, contrary to what intuition should expect. Enabling optimisations should been seen
as a request to make the user's code faster, removing `debug_assert!` and other checks seems like
a natural consequence.

# Detailed design

The `debug_assertions` configuration variable, the replacement for the `ndebug` variable, will be
compiler provided based on the value of the `opt-level` codegen flag, including the implied value
from `-O`.  Any value higher than 0 will disable the variable.

Another codegen flag `debug-assertions` will override this, forcing it on or off based on the value
passed to it.

# Drawbacks

Technically backwards incompatible change. However the only usage of the `ndebug` variable in the
rust tree is in the implementation of `debug_assert!`, so it's unlikely that any external code is
using it.

# Alternatives

No real alternatives beyond different names and defaults.

# Unresolved questions

None.