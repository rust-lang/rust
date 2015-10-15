% Lifetime Elision

In order to make common patterns more ergonomic, Rust allows lifetimes to be
*elided* in function signatures.

A *lifetime position* is anywhere you can write a lifetime in a type:

```rust,ignore
&'a T
&'a mut T
T<'a>
```

Lifetime positions can appear as either "input" or "output":

* For `fn` definitions, input refers to the types of the formal arguments
  in the `fn` definition, while output refers to
  result types. So `fn foo(s: &str) -> (&str, &str)` has elided one lifetime in
  input position and two lifetimes in output position.
  Note that the input positions of a `fn` method definition do not
  include the lifetimes that occur in the method's `impl` header
  (nor lifetimes that occur in the trait header, for a default method).

* In the future, it should be possible to elide `impl` headers in the same manner.

Elision rules are as follows:

* Each elided lifetime in input position becomes a distinct lifetime
  parameter.

* If there is exactly one input lifetime position (elided or not), that lifetime
  is assigned to *all* elided output lifetimes.

* If there are multiple input lifetime positions, but one of them is `&self` or
  `&mut self`, the lifetime of `self` is assigned to *all* elided output lifetimes.

* Otherwise, it is an error to elide an output lifetime.

Examples:

```rust,ignore
fn print(s: &str);                                      // Elided.
fn print<'a>(s: &'a str);                               // Expanded.

fn debug(lvl: uint, s: &str);                           // Elided.
fn debug<'a>(lvl: uint, s: &'a str);                    // Expanded.

fn substr(s: &str, until: uint) -> &str;                // Elided.
fn substr<'a>(s: &'a str, until: uint) -> &'a str;      // Expanded.

fn get_str() -> &str;                                   // ILLEGAL.

fn frob(s: &str, t: &str) -> &str;                      // ILLEGAL.

fn get_mut(&mut self) -> &mut T;                        // Elided.
fn get_mut<'a>(&'a mut self) -> &'a mut T;              // Expanded.

fn args<T:ToCStr>(&mut self, args: &[T]) -> &mut Command                  // Elided.
fn args<'a, 'b, T:ToCStr>(&'a mut self, args: &'b [T]) -> &'a mut Command // Expanded.

fn new(buf: &mut [u8]) -> BufWriter;                    // Elided.
fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a>          // Expanded.

```
