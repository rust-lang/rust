- Start Date: 2014-05-04
- RFC PR: [rust-lang/rfcs#66](https://github.com/rust-lang/rfcs/pull/66)
- Rust Issue: [rust-lang/rust#15023](https://github.com/rust-lang/rust/issues/15023)

# Summary

Temporaries live for the enclosing block when found in a let-binding. This only
holds when the reference to the temporary is taken directly. This logic should
be extended to extend the cleanup scope of any temporary whose lifetime ends up
in the let-binding.

For example, the following doesn't work now, but should:

```rust
use std::os;

fn main() {
	let x = os::args().slice_from(1);
	println!("{}", x);
}
```

# Motivation

Temporary lifetimes are a bit confusing right now. Sometimes you can keep
references to them, and sometimes you get the dreaded "borrowed value does not
live long enough" error. Sometimes one operation works but an equivalent
operation errors, e.g. autoref of `~[T]` to `&[T]` works but calling
`.as_slice()` doesn't. In general it feels as though the compiler is simply
being overly restrictive when it decides the temporary doesn't live long
enough.

# Drawbacks

I can't think of any drawbacks.

# Detailed design

When a reference to a temporary is passed to a function (either as a regular
argument or as the `self` argument of a method), and the function returns a
value with the same lifetime as the temporary reference, the lifetime of the
temporary should be extended the same way it would if the function was not
invoked.

For example, `~[T].as_slice()` takes `&'a self` and returns `&'a [T]`. Calling
`as_slice()` on a temporary of type `~[T]` will implicitly take a reference
`&'a ~[T]` and return a value `&'a [T]` This return value should be considered
to extend the lifetime of the `~[T]` temporary just as taking an explicit
reference (and skipping the method call) would.

# Alternatives

Don't do this. We live with the surprising borrowck errors and the ugly workarounds that look like

```rust
let x = os::args();
let x = x.slice_from(1);
```

# Unresolved questions

None that I know of.
