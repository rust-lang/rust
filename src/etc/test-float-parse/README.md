# Float Parsing Tests

These are tests designed to test decimal to float conversions (`dec2flt`) used
by the standard library.

It consists of a collection of test generators that each generate a set of
patterns intended to test a specific property. In addition, there are exhaustive
tests (for <= `f32`) and fuzzers (for anything that can't be run exhaustively).

The generators work as follows:

- Each generator is a struct that lives somewhere in the `gen` module. Usually
  it is generic over a float type.
- These generators must implement `Iterator`, which should return a context type
  that can be used to construct a test string (but usually not the string
  itself).
- They must also implement the `Generator` trait, which provides a method to
  write test context to a string as a test case, as well as some extra metadata.

  The split between context generation and string construction is so that we can
  reuse string allocations.
- Each generator gets registered once for each float type. Each of these
  generators then get their iterator called, and each test case checked against
  the float type's parse implementation.

Some generators produce decimal strings, others create bit patterns that need to
be bitcasted to the float type, which then uses its `Display` implementation to
write to a string. For these, float to decimal (`flt2dec`) conversions also get
tested, if unintentionally.

For each test case, the following is done:

- The test string is parsed to the float type using the standard library's
  implementation.
- The test string is parsed separately to a `BigRational`, which acts as a
  representation with infinite precision.
- The rational value then gets checked that it is within the float's
  representable values (absolute value greater than the smallest number to round
  to zero, but less less than the first value to round to infinity). If these
  limits are exceeded, check that the parsed float reflects that.
- For real nonzero numbers, the parsed float is converted into a rational using
  `significand * 2^exponent`. It is then checked against the actual rational
  value, and verified to be within half a bit's precision of the parsed value.
  Also it is checked that ties round to even.

This is all highly parallelized with `rayon`; test generators can run in
parallel, and their tests get chunked and run in parallel.

There is a simple command line that allows filtering which tests are run,
setting the number of iterations for fuzzing tests, limiting failures, setting
timeouts, etc. See `main.rs` or run with `--help` for options.

Note that when running via `./x`, only tests that take less than a few minutes
are run by default. Navigate to the crate (or pass `-C` to Cargo) and run it
directly to run all tests or pass specific arguments.
