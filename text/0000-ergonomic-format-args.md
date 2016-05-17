- Feature Name: `ergonomic_format_args`
- Start Date: 2016-05-17
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Removes the one-type-only restriction on `format_args!` arguments.

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

## Overview

Formatting is done during both compile-time (expansion-time to be pedantic)
and runtime in Rust. As we are concerned with format string parsing, not
outputting, this RFC only touches the compile-time side of the existing
formatting mechanism which is `libsyntax_ext` and `libfmt_macros`.

Before continuing with the details, it is worth noting that the core flow of
current Rust formatting is *mapping arguments to placeholders to format specs*.
For clarity, we distinguish among *placeholders*, *macro arguments* and
*generated `ArgumentV1` objects*. They are all *italicized* to provide some
visual hint for distinction.

To implement the proposed design, first we resolve all implicit references to
the next argument (*next-references* for short) during parse; then we modify
the macro expansion to make use of the now explicit argument references,
preserving the mapping.

## Parse-time next-reference resolution

Currently two forms of next-references exist: `ArgumentNext` and
`CountIsNextParam`. Both take a positional *macro argument* and advance the
same internal pointer, but format is parsed before position, as shown in
format strings like `"{foo:.*} {} {:.*}"` which is in every way equivalent to
`"{foo:.0$} {1} {3:.2$}"`.

As the rule is already known even at compile-time, and does not require the
whole format string to be known beforehand, the resolution can happen just
inside the parser after a *placeholder* is successfully parsed. As a natural
consequence, both forms of next-reference can be removed from the rest of the
compiler, simplifying work later.

## Expansion-time argument mapping

There are two kinds of *macro arguments*, positional and named. Because of the
apparent type difference, two maps are needed to track *placeholder* types
(known as `ArgumentType`s in the code). In the current implementation,
`Vec<Option<ArgumentType>>` is for positional *macro arguments* and
`HashMap<String, ArgumentType>` is for named *macro arguments*, apparently
neither of which supports multiple types for one *macro argument*. Also, for
constructing the `__STATIC_FMTARGS` we need to first figure out the order for
every *placeholder* in the list of *generated `ArgumentV1` objects*. So we
first classify *placeholders* according to their associated *macro arguments*,
which are all explicit now, then assign each of them a correct index.

### Placeholder type collection

In the proposed design, lists of `ArgumentType`s are used to store
*placeholder* types for each *macro argument* in order. During verification
the *placeholder* type seen for a *macro argument* is simply pushed into the
respective list. This does not remove the ability to sense unused
*macro arguments*, as the list would simply be empty when checked later, just
as it would be `None` in the old `Option<ArgumentType>` version.

### Mapping construction

For consistency with the current implementation, named *macro arguments* are
still put at the end of *generated `ArgumentV1` objects*. Which means we have
to consume all of format string in order to know how many *placeholders* there
are referencing to positional *macro arguments*. As such, the verification
and translation of pieces are now separated with mapping construction in
between.

Obviously, the orders used during mapping and actual expansion must agree, but
fortunately the rules are very simple now only explicit references remain.
We iterate over the list of known positional *macro arguments*, recording the
index at which every bunch of *generated `ArgumentV1` objects* would begin for
each positional *macro argument*. After that, we also record the total number
for mapping the named *macro arguments*, as the relative offsets of named
*placeholders* are already recorded during verification.

### Expansion

With mapping between *placeholders* and *generated `ArgumentV1` objects*
ready at hand, it is easy to emit correct `Argument`s. Scratch space is
provided to `trans_piece` for remembering how many *placeholders* for a given
*macro argument* have been processed. This information is then used to rewrite
all references from using *macro argument* indices to
*generated `ArgumentV1` object* indices, namely:

* `ArgumentIs(i)`
* `ArgumentNamed(n)`
* `CountIsParam(i)`
* `CountIsName(n)`

For the count references, some may suggest that they are now potentially
ambiguous. However considering the implementation of `verify_count`, the
parameter used by each `Count` is individually injected into the list of
*generated `ArgumentV1` objects* as if it were explicitly specified. Also it
is *macro arguments* to be referenced, not the potentially multiple
*placeholders*, so there are in fact no ambiguities.

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

* Does the *generated `ArgumentV1` objects* need deduplication?
* Will it break the ABI if handling of next-references in `libcore/fmt` is removed as well?
