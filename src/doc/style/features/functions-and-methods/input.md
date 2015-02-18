% Input to functions and methods

### Let the client decide when to copy and where to place data. [FIXME: needs RFC]

#### Copying:

Prefer

```rust
fn foo(b: Bar) {
   // use b as owned, directly
}
```

over

```rust
fn foo(b: &Bar) {
    let b = b.clone();
    // use b as owned after cloning
}
```

If a function requires ownership of a value of unknown type `T`, but does not
otherwise need to make copies, the function should take ownership of the
argument (pass by value `T`) rather than using `.clone()`. That way, the caller
can decide whether to relinquish ownership or to `clone`.

Similarly, the `Copy` trait bound should only be demanded it when absolutely
needed, not as a way of signaling that copies should be cheap to make.

#### Placement:

Prefer

```rust
fn foo(b: Bar) -> Bar { ... }
```

over

```rust
fn foo(b: Box<Bar>) -> Box<Bar> { ... }
```

for concrete types `Bar` (as opposed to trait objects). This way, the caller can
decide whether to place data on the stack or heap. No overhead is imposed by
letting the caller determine the placement.

### Minimize assumptions about parameters. [FIXME: needs RFC]

The fewer assumptions a function makes about its inputs, the more widely usable
it becomes.

#### Minimizing assumptions through generics:

Prefer

```rust
fn foo<T: Iterator<int>>(c: T) { ... }
```

over any of

```rust
fn foo(c: &[int]) { ... }
fn foo(c: &Vec<int>) { ... }
fn foo(c: &SomeOtherCollection<int>) { ... }
```

if the function only needs to iterate over the data.

More generally, consider using generics to pinpoint the assumptions a function
needs to make about its arguments.

On the other hand, generics can make it more difficult to read and understand a
function's signature. Aim for "natural" parameter types that a neither overly
concrete nor overly abstract. See the discussion on
[traits](../../traits/README.md) for more guidance.


#### Minimizing ownership assumptions:

Prefer either of

```rust
fn foo(b: &Bar) { ... }
fn foo(b: &mut Bar) { ... }
```

over

```rust
fn foo(b: Bar) { ... }
```

That is, prefer borrowing arguments rather than transferring ownership, unless
ownership is actually needed.

### Prefer compound return types to out-parameters. [FIXME: needs RFC]

Prefer

```rust
fn foo() -> (Bar, Bar)
```

over

```rust
fn foo(output: &mut Bar) -> Bar
```

for returning multiple `Bar` values.

Compound return types like tuples and structs are efficiently compiled
and do not require heap allocation. If a function needs to return
multiple values, it should do so via one of these types.

The primary exception: sometimes a function is meant to modify data
that the caller already owns, for example to re-use a buffer:

```rust
fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>
```

(From the [Reader trait](http://static.rust-lang.org/doc/master/std/io/trait.Reader.html#tymethod.read).)

### Consider validating arguments, statically or dynamically. [FIXME: needs RFC]

_Note: this material is closely related to
  [library-level guarantees](../../safety/lib-guarantees.md)._

Rust APIs do _not_ generally follow the
[robustness principle](http://en.wikipedia.org/wiki/Robustness_principle): "be
conservative in what you send; be liberal in what you accept".

Instead, Rust code should _enforce_ the validity of input whenever practical.

Enforcement can be achieved through the following mechanisms (listed
in order of preference).

#### Static enforcement:

Choose an argument type that rules out bad inputs.

For example, prefer

```rust
fn foo(a: ascii::Ascii) { ... }
```

over

```rust
fn foo(a: u8) { ... }
```

Note that
[`ascii::Ascii`](http://static.rust-lang.org/doc/master/std/ascii/struct.Ascii.html)
is a _wrapper_ around `u8` that guarantees the highest bit is zero; see
[newtype patterns]() for more details on creating typesafe wrappers.

Static enforcement usually comes at little run-time cost: it pushes the
costs to the boundaries (e.g. when a `u8` is first converted into an
`Ascii`). It also catches bugs early, during compilation, rather than through
run-time failures.

On the other hand, some properties are difficult or impossible to
express using types.

#### Dynamic enforcement:

Validate the input as it is processed (or ahead of time, if necessary).  Dynamic
checking is often easier to implement than static checking, but has several
downsides:

1. Runtime overhead (unless checking can be done as part of processing the input).
2. Delayed detection of bugs.
3. Introduces failure cases, either via `fail!` or `Result`/`Option` types (see
   the [error handling guidelines](../../errors/README.md)), which must then be
   dealt with by client code.

#### Dynamic enforcement with `debug_assert!`:

Same as dynamic enforcement, but with the possibility of easily turning off
expensive checks for production builds.

#### Dynamic enforcement with opt-out:

Same as dynamic enforcement, but adds sibling functions that opt out of the
checking.

The convention is to mark these opt-out functions with a suffix like
`_unchecked` or by placing them in a `raw` submodule.

The unchecked functions can be used judiciously in cases where (1) performance
dictates avoiding checks and (2) the client is otherwise confident that the
inputs are valid.

> **[FIXME]** Should opt-out functions be marked `unsafe`?
