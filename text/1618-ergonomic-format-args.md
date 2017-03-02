- Feature Name: (not applicable)
- Start Date: 2016-05-17
- RFC PR: [rust-lang/rfcs#1618](https://github.com/rust-lang/rfcs/pull/1618)
- Rust Issue: [rust-lang/rust#33642](https://github.com/rust-lang/rust/pull/33642)

# Summary
[summary]: #summary

Removes the one-type-only restriction on `format_args!` arguments.
Expressions like `format_args!("{0:x} {0:o}", foo)` now work as intended,
where each argument is still evaluated only once, in order of appearance
(i.e. left-to-right).

# Motivation
[motivation]: #motivation

The `format_args!` macro and its friends historically only allowed a single
type per argument, such that trivial format strings like `"{0:?} == {0:x}"` or
`"rgb({r}, {g}, {b}) is #{r:02x}{g:02x}{b:02x}"` are illegal. This is
massively inconvenient and counter-intuitive, especially considering the
formatting syntax is borrowed from Python where such things are perfectly
valid.

Upon closer investigation, the restriction is in fact an artificial
implementation detail. For mapping format placeholders to macro arguments the
`format_args!` implementation did not bother to record type information for
all the placeholders sequentially, but rather chose to remember only one type
per argument. Also the formatting logic has not received significant attention
since after its conception, but the uses have greatly expanded over the years,
so the mechanism as a whole certainly needs more love.

# Detailed design
[design]: #detailed-design

Formatting is done during both compile-time (expansion-time to be pedantic)
and runtime in Rust. As we are concerned with format string parsing, not
outputting, this RFC only touches the compile-time side of the existing
formatting mechanism which is `libsyntax_ext` and `libfmt_macros`.

Before continuing with the details, it is worth noting that the core flow of
current Rust formatting is *mapping arguments to placeholders to format specs*.
For clarity, we distinguish among *placeholders*, *macro arguments* and
*argument objects*. They are all *italicized* to provide some
visual hint for distinction.

To implement the proposed design, the following changes in behavior are made:

* implicit references are resolved during parse of format string;
* named *macro arguments* are resolved into positional ones;
* placeholder types are remembered and de-duplicated for each *macro argument*,
* the *argument objects* are emitted with information gathered in steps above.

As most of the details is best described in the code itself, we only
illustrate some of the high-level changes below.

## Implicit reference resolution

Currently two forms of implicit references exist: `ArgumentNext` and
`CountIsNextParam`. Both take a positional *macro argument* and advance the
same internal pointer, but format is parsed before position, as shown in
format strings like `"{foo:.*} {} {:.*}"` which is in every way equivalent to
`"{foo:.0$} {1} {3:.2$}"`.

As the rule is already known even at compile-time, and does not require the
whole format string to be known beforehand, the resolution can happen just
inside the parser after a *placeholder* is successfully parsed. As a natural
consequence, both forms can be removed from the rest of the compiler,
simplifying work later.

## Named argument resolution

Not seen elsewhere in Rust, named arguments in format macros are best seen as
syntactic sugar, and we'd better actually treat them as such. Just after
successfully parsing the *macro arguments*, we immediately rewrite every name
to its respective position in the argument list, which again simplifies the
process.

## Processing and expansion

We only have absolute positional references to *macro arguments* at this point,
and it's straightforward to remember all unique *placeholders* encountered for
each. The unique *placeholders* are emitted into *argument objects* in order,
to preserve evaluation order, but no difference in behavior otherwise.

# Drawbacks
[drawbacks]: #drawbacks

Due to the added data structures and processing, time and memory costs of
compilations may slightly increase. However this is mere speculation without
actual profiling and benchmarks. Also the ergonomical benefits alone justifies
the additional costs.

# Alternatives
[alternatives]: #alternatives

## Do nothing

One can always write a little more code to simulate the proposed behavior,
and this is what people have most likely been doing under today's constraints.
As in:

```rust
fn main() {
	let r = 0x66;
	let g = 0xcc;
	let b = 0xff;

	// rgb(102, 204, 255) == #66ccff
	// println!("rgb({r}, {g}, {b}) == #{r:02x}{g:02x}{b:02x}", r=r, g=g, b=b);
	println!("rgb({}, {}, {}) == #{:02x}{:02x}{:02x}", r, g, b, r, g, b);
}
```

Or slightly more verbose when side effects are in play:

```rust
fn do_something(i: &mut usize) -> usize {
	let result = *i;
	*i += 1;
	result
}

fn main() {
	let mut i = 0x1234usize;

	// 0b1001000110100 0o11064 0x1234
	// 0x1235
	// println!("{0:#b} {0:#o} {0:#x}", do_something(&mut i));
	// println!("{:#x}", i);

	// need to consider side effects, hence a temp var
	{
		let r = do_something(&mut i);
		println!("{:#b} {:#o} {:#x}", r, r, r);
		println!("{:#x}", i);
	}
}
```

While the effects are the same and nothing requires modification, the
ergonomics is simply bad and the code becomes unnecessarily convoluted.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
