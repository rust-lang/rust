- Feature Name: `rustc_macros`
- Start Date: 2016-07-14
- RFC PR: https://github.com/rust-lang/rfcs/pull/1681
- Rust Issue: https://github.com/rust-lang/rust/issues/35900

# Summary
[summary]: #summary

Extract a very small sliver of today's procedural macro system in the compiler,
just enough to get basic features like custom derive working, to have an
eventually stable API. Ensure that these features will not pose a maintenance
burden on the compiler but also don't try to provide enough features for the
"perfect macro system" at the same time. Overall, this should be considered an
incremental step towards an official "macros 2.0".

# Motivation
[motivation]: #motivation

Some large projects in the ecosystem today, such as [serde] and [diesel],
effectively require the nightly channel of the Rust compiler. Although most
projects have an alternative to work on stable Rust, this tends to be far less
ergonomic and comes with its own set of downsides, and empirically it has not
been enough to push the nightly users to stable as well.

[serde]: https://github.com/serde-rs/serde
[diesel]: http://diesel.rs/

These large projects, however, are often the face of Rust to external users.
Common knowledge is that fast serialization is done using serde, but to others
this just sounds like "fast Rust needs nightly". Over time this persistent
thought process creates a culture of "well to be serious you require nightly"
and a general feeling that Rust is not "production ready".

The good news, however, is that this class of projects which require nightly
Rust almost all require nightly for the reason of procedural macros. Even
better, the full functionality of procedural macros is rarely needed, only
custom derive! Even better, custom derive typically doesn't *require* the features
one would expect from a full-on macro system, such as hygiene and modularity,
that normal procedural macros typically do. The purpose of this RFC, as a
result, is to provide these crates a method of working on stable Rust with the
desired ergonomics one would have on nightly otherwise.

Unfortunately today's procedural macros are not without their architectural
shortcomings as well. For example they're defined and imported with arcane
syntax and don't participate in hygiene very well. To address these issues,
there are a number of RFCs to develop a "macros 2.0" story:

* [Changes to name resolution](https://github.com/rust-lang/rfcs/pull/1560)
* [Macro naming and modularisation](https://github.com/rust-lang/rfcs/pull/1561)
* [Procedural macros](https://github.com/rust-lang/rfcs/pull/1566)
* [Macros by example 2.0](https://github.com/rust-lang/rfcs/pull/1584)

Many of these designs, however, will require a significant amount of work to not
only implement but also a significant amount of work to stabilize. The current
understanding is that these improvements are on the time scale of years, whereas
the problem of nightly Rust is today!

As a result, it is an explicit non-goal of this RFC to architecturally improve
on the current procedural macro system. The drawbacks of today's procedural
macros will be the same as those proposed in this RFC. The major goal here is
to simply minimize the exposed surface area between procedural macros and the
compiler to ensure that the interface is well defined and can be stably
implemented in future versions of the compiler as well.

Put another way, we currently have macros 1.0 unstable today, we're shooting
for macros 2.0 stable in the far future, but this RFC is striking a middle
ground at macros 1.1 today!

# Detailed design
[design]: #detailed-design

First, before looking how we're going to expose procedural macros, let's
take a detailed look at how they work today.

### Today's procedural macros

A procedural macro today is loaded into a crate with the `#![plugin(foo)]`
annotation at the crate root. This in turn looks for a crate named `foo` [via
the same crate loading mechanisms][loader] as `extern crate`, except [with the
restriction][host-restriction] that the target triple of the crate must be the
same as the target the compiler was compiled for. In other words, if you're on
x86 compiling to ARM, macros must also be compiled for x86.

[loader]: https://github.com/rust-lang/rust/blob/78d49bfac2bbcd48de522199212a1209f498e834/src/librustc_metadata/creader.rs#L480
[host-restriction]: https://github.com/rust-lang/rust/blob/78d49bfac2bbcd48de522199212a1209f498e834/src/librustc_metadata/creader.rs#L494

Once a crate is found, it's required to be a dynamic library as well, and once
that's all verified the compiler [opens it up with `dlopen`][dlopen] (or the
equivalent therein). After loading, the compiler will [look for a special
symbol][symbol] in the dynamic library, and then call it with a macro context.

[dlopen]: https://github.com/rust-lang/rust/blob/78d49bfac2bbcd48de522199212a1209f498e834/src/librustc_plugin/load.rs#L124
[symbol]: https://github.com/rust-lang/rust/blob/78d49bfac2bbcd48de522199212a1209f498e834/src/librustc_plugin/load.rs#L136-L139

So as we've seen macros are compiled as normal crates into dynamic libraries.
One function in the crate is tagged with `#[plugin_registrar]` which gets wired
up to this "special symbol" the compiler wants. When the function is called with
a macro context, it uses the passed in [plugin registry][registry] to register
custom macros, attributes, etc.

[registry]: https://github.com/rust-lang/rust/blob/78d49bfac2bbcd48de522199212a1209f498e834/src/librustc_plugin/registry.rs#L30-L69

After a macro is registered, the compiler will then continue the normal process
of expanding a crate. Whenever the compiler encounters this macro it will call
this registration with essentially and AST and morally gets back a different
AST to splice in or replace.

### Today's drawbacks

This expansion process suffers from many of the downsides mentioned in the
motivation section, such as a lack of hygiene, a lack of modularity, and the
inability to import macros as you would normally other functionality in the
module system.

Additionally, though, it's essentially impossible to ever *stabilize* because
the interface to the compiler is... the compiler! We clearly want to make
changes to the compiler over time, so this isn't acceptable. To have a stable
interface we'll need to cut down this surface area *dramatically* to a curated
set of known-stable APIs.

Somewhat more subtly, the technical ABI of procedural macros is also exposed
quite thinly today as well. The implementation detail of dynamic libraries, and
especially that both the compiler and the macro dynamically link to libraries
like libsyntax, cannot be changed. This precludes, for example, a completely
statically linked compiler (e.g. compiled for `x86_64-unknown-linux-musl`).
Another goal of this RFC will also be to hide as many of these technical
details as possible, allowing the compiler to flexibly change how it interfaces
to macros.

## Macros 1.1

Ok, with the background knowledge of what procedural macros are today, let's
take a look at how we can solve the major problems blocking its stabilization:

* Sharing an API of the entire compiler
* Frozen interface between the compiler and macros

### `librustc_macro`

Proposed in [RFC 1566](https://github.com/rust-lang/rfcs/pull/1566) and
described in [this blog post](http://ncameron.org/blog/libmacro/) the
distribution will now ship with a new `librustc_macro` crate available for macro
authors. The intention here is that the gory details of how macros *actually*
talk to the compiler is entirely contained within this one crate. The stable
interface to the compiler is then entirely defined in this crate, and we can
make it as small or large as we want. Additionally, like the standard library,
it can contain unstable APIs to test out new pieces of functionality over time.

The initial implementation of `librustc_macro` is proposed to be *incredibly*
bare bones:

```rust
#![crate_name = "macro"]

pub struct TokenStream {
    // ...
}

#[derive(Debug)]
pub struct LexError {
    // ...
}

impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(s: &str) -> Result<TokenStream, LexError> {
        // ...
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // ...
    }
}
```

That is, there will only be a handful of exposed types and `TokenStream` can
only be converted to and from a `String`. Eventually `TokenStream` type will
more closely resemble token streams [in the compiler
itself][compiler-tokenstream], and more fine-grained manipulations will be
available as well.

[compiler-tokenstream]: https://github.com/rust-lang/rust/blob/master/src/libsyntax/tokenstream.rs#L323-L338

### Defining a macro

A new crate type will be added to the compiler, `rustc-macro` (described below),
indicating a crate that's compiled as a procedural macro. There will not be a
"registrar" function in this crate type (like there is today), but rather a
number of functions which act as token stream transformers to implement macro
functionality.

A macro crate might look like:

```rust
#![crate_type = "rustc-macro"]
#![crate_name = "double"]

extern crate rustc_macro;

use rustc_macro::TokenStream;

#[rustc_macro_derive(Double)]
pub fn double(input: TokenStream) -> TokenStream {
    let source = input.to_string();

    // Parse `source` for struct/enum declaration, and then build up some new
    // source code representing a number of items in the implementation of
    // the `Double` trait for the struct/enum in question.
    let source = derive_double(&source);

    // Parse this back to a token stream and return it
    source.parse().unwrap()
}
```

This new `rustc_macro_derive` attribute will be allowed inside of a
`rustc-macro` crate but disallowed in other crate types. It defines a new
`#[derive]` mode which can be used in a crate. The input here is the entire
struct that `#[derive]` was attached to, attributes and all. The output is
**expected to include the `struct`/`enum` itself** as well as any number of
items to be contextually "placed next to" the initial declaration.

Again, though, there is no hygiene. More specifically, the
`TokenStream::from_str` method will use the same expansion context as the derive
attribute itself, not the point of definition of the derive function. All span
information for the `TokenStream` structures returned by `from_source` will
point to the original `#[derive]` annotation. This means that error messages
related to struct definitions will get *worse* if they have a custom derive
attribute placed on them, because the entire struct's span will get folded into
the `#[derive]` annotation. Eventually, though, more span information will be
stable on the `TokenStream` type, so this is just a temporary limitation.

The `rustc_macro_derive` attribute requires the signature (similar to [macros
2.0][mac20sig]):

[mac20sig]: http://ncameron.org/blog/libmacro/#tokenisingandquasiquoting

```rust
fn(TokenStream) -> TokenStream
```

If a macro cannot process the input token stream, it is expected to panic for
now, although eventually it will call methods in `rustc_macro` to provide more
structured errors. The compiler will wrap up the panic message and display it
to the user appropriately. Eventually, however, `librustc_macro` will provide
more interesting methods of signaling errors to users.

Customization of user-defined `#[derive]` modes can still be done through custom
attributes, although it will be required for `rustc_macro_derive`
implementations to remove these attributes when handing them back to the
compiler. The compiler will still gate unknown attributes by default.

### `rustc-macro` crates

Like the rlib and dylib crate types, the `rustc-macro` crate
type is intended to be an intermediate product.  What it *actually* produces is
not specified, but if a `-L` path is provided to it then the compiler will
recognize the output artifacts as a macro and it can be loaded for a program.

Initially if a crate is compiled with the `rustc-macro` crate type (and possibly
others) it will forbid exporting any items in the crate other than those
functions tagged `#[rustc_macro_derive]` and those functions must also be placed
at the crate root. Finally, the compiler will automatically set the
`cfg(rustc_macro)` annotation whenever any crate type of a compilation is the
`rustc-macro` crate type.

While these properties may seem a bit odd, they're intended to allow a number of
forwards-compatible extensions to be implemented in macros 2.0:

* Macros eventually want to be imported from crates (e.g. `use foo::bar!`) and
  limiting where `#[derive]` can be defined reduces the surface area for
  possible conflict.
* Macro crates eventually want to be compiled to be available both at runtime
  and at compile time. That is, an `extern crate foo` annotation may load
  *both* a `rustc-macro` crate and a crate to link against, if they are
  available. Limiting the public exports for now to only custom-derive
  annotations should allow for maximal flexibility here.

### Using a procedural macro

Using a procedural macro will be very similar to today's `extern crate` system,
such as:

```rust
#[macro_use]
extern crate double;

#[derive(Double)]
pub struct Foo;

fn main() {
    // ...
}
```

That is, the `extern crate` directive will now also be enhanced to look for
crates compiled as `rustc-macro` in addition to those compiled as `dylib` and
`rlib`. Today this will be temporarily limited to finding *either* a
`rustc-macro` crate or an rlib/dylib pair compiled for the target, but this
restriction may be lifted in the future.

The custom derive annotations loaded from `rustc-macro` crates today will all be
placed into the same global namespace. Any conflicts (shadowing) will cause the
compiler to generate an error, and it must be resolved by loading only one or
the other of the `rustc-macro` crates (eventually this will be solved with a
more principled `use` system in macros 2.0).

### Initial implementation details

This section lays out what the initial implementation details of macros 1.1
will look like, but none of this will be specified as a stable interface to the
compiler. These exact details are subject to change over time as the
requirements of the compiler change, and even amongst platforms these details
may be subtly different.

The compiler will essentially consider `rustc-macro` crates as `--crate-type
dylib -C prefer-dyanmic`. That is, compiled the same way they are today. This
namely means that these macros  will dynamically link to the same standard
library as the compiler itself, therefore sharing resources like a global
allocator, etc.

The `librustc_macro` crate will compiled as an rlib and a static copy of it
will be included in each macro. This crate will provide a symbol known by the
compiler that can be dynamically loaded. The compiler will `dlopen` a macro
crate in the same way it does today, find this symbol in `librustc_macro`, and
call it.

The `rustc_macro_derive` attribute will be encoded into the crate's metadata,
and the compiler will discover all these functions, load their function
pointers, and pass them to the `librustc_macro` entry point as well. This
provides the opportunity to register all the various expansion mechanisms with
the compiler.

The actual underlying representation of `TokenStream` will be basically the same
as it is in the compiler today. (the details on this are a little light
intentionally, shouldn't be much need to go into *too* much detail).

### Initial Cargo integration

Like plugins today, Cargo needs to understand which crates are `rustc-macro`
crates and which aren't. Cargo additionally needs to understand this to sequence
compilations correctly and ensure that `rustc-macro` crates are compiled for the
host platform. To this end, Cargo will understand a new attribute in the `[lib]`
section:

```toml
[lib]
rustc-macro = true
```

This annotation indicates that the crate being compiled should be compiled as a
`rustc-macro` crate type for the host platform in the current compilation.

Eventually Cargo may also grow support to understand that a `rustc-macro` crate
should be compiled twice, once for the host and once for the target, but this is
intended to be a backwards-compatible extension to Cargo.

## Pieces to stabilize

Eventually this RFC is intended to be considered for stabilization (after it's
implemented and proven out on nightly, of course). The summary of pieces that
would become stable are:

* The `rustc_macro` crate, and a small set of APIs within (skeleton above)
* The `rustc-macro` crate type, in addition to its current limitations
* The `#[rustc_macro_derive]` attribute
* The signature of the `#![rustc_macro_derive]` functions
* Semantically being able to load macro crates compiled as `rustc-macro` into
  the compiler, requiring that the crate was compiled by the exact compiler.
* The semantic behavior of loading custom derive annotations, in that they're
  just all added to the same global namespace with errors on conflicts.
  Additionally, definitions end up having no hygiene for now.
* The `rustc-macro = true` attribute in Cargo

### Macros 1.1 in practice

Alright, that's a lot to take in! Let's take a look at what this is all going to
look like in practice, focusing on a case study of `#[derive(Serialize)]` for
serde.

First off, serde will provide a crate, let's call it `serde_macros`. The
`Cargo.toml` will look like:

```toml
[package]
name = "serde-macros"
# ...

[lib]
rustc-macro = true

[dependencies]
syntex_syntax = "0.38.0"
```

The contents will look similar to

```rust
extern crate rustc_macro;
extern crate syntex_syntax;

use rustc_macro::TokenStream;

#[rustc_macro_derive(Serialize)]
pub fn derive_serialize(input: TokenStream) -> TokenStream {
    let input = input.to_string();

    // use syntex_syntax from crates.io to parse `input` into an AST

    // use this AST to generate an impl of the `Serialize` trait for the type in
    // question

    // convert that impl to a string

    // parse back into a token stream
    return impl_source.parse().unwrap()
}
```

Next, crates will depend on this such as:

```toml
[dependencies]
serde = "0.9"
serde-macros = "0.9"
```

And finally use it as such:

```rust
extern crate serde;
#[macro_use]
extern crate serde_macros;

#[derive(Serialize)]
pub struct Foo {
    a: usize,
    #[serde(rename = "foo")]
    b: String,
}
```

# Drawbacks
[drawbacks]: #drawbacks

* This is not an interface that would be considered for stabilization in a void,
  there are a number of known drawbacks to the current macro system in terms of
  how it architecturally fits into the compiler. Additionally, there's work
  underway to solve all these problems with macros 2.0.

  As mentioned before, however, the stable version of macros 2.0 is currently
  quite far off, and the desire for features like custom derive are very real
  today. The rationale behind this RFC is that the downsides are an acceptable
  tradeoff from moving a significant portion of the nightly ecosystem onto stable
  Rust.

* This implementation is likely to be less performant than procedural macros
  are today. Round tripping through strings isn't always a speedy operation,
  especially for larger expansions. Strings, however, are a very small
  implementation detail that's easy to see stabilized until the end of time.
  Additionally, it's planned to extend the `TokenStream` API in the future to
  allow more fine-grained transformations without having to round trip through
  strings.

* Users will still have an inferior experience to today's nightly macros
  specifically with respect to compile times. The `syntex_syntax` crate takes
  quite a few seconds to compile, and this would be required by any crate which
  uses serde. To offset this, though, the `syntex_syntax` could be *massively*
  stripped down as all it needs to do is parse struct declarations mostly. There
  are likely many other various optimizations to compile time that can be
  applied to ensure that it compiles quickly.

* Plugin authors will need to be quite careful about the code which they
  generate as working with strings loses much of the expressiveness of macros in
  Rust today. For example:

  ```rust
  macro_rules! foo {
      ($x:expr) => {
          #[derive(Serialize)]
          enum Foo { Bar = $x, Baz = $x * 2 }
      }
  }
  foo!(1 + 1);
  ```

  Plugin authors would have to ensure that this is not naively interpreted as
  `Baz = 1 + 1 * 2` as this will cause incorrect results. The compiler will also
  need to be careful to parenthesize token streams like this when it generates
  a stringified source.

* By having separte library and macro crate support today (e.g. `serde` and
  `serde_macros`) it's possible for there to be version skew between the two,
  making it tough to ensure that the two versions you're using are compatible
  with one another. This would be solved if `serde` itself could define or
  reexport the macros, but unfortunately that would require a likely much larger
  step towards "macros 2.0" to solve and would greatly increase the size of this
  RFC.
  
* Converting to a string and back loses span information, which can
  lead to degraded error messages. For example, currently we can make
  an effort to use the span of a given field when deriving code that
  is caused by that field, but that kind of precision will not be
  possible until a richer interface is available.

# Alternatives
[alternatives]: #alternatives

* Wait for macros 2.0, but this likely comes with the high cost of postponing a
  stable custom-derive experience on the time scale of years.

* Don't add `rustc_macro` as a new crate, but rather specify that
  `#[rustc_macro_derive]` has a stable-ABI friendly signature. This does not
  account, however, for the eventual planned introduction of the `rustc_macro`
  crate and is significantly harder to write. The marginal benefit of being
  slightly more flexible about how it's run likely isn't worth it.

* The syntax for defining a macro may be different in the macros 2.0 world (e.g.
  `pub macro foo` vs an attribute), that is it probably won't involve a function
  attribute like `#[rustc_macro_derive]`. This interim system could possibly use
  this syntax as well, but it's unclear whether we have a concrete enough idea
  in mind to implement today.

* The `TokenStream` state likely has some sort of backing store behind it like a
  string interner, and in the APIs above it's likely that this state is passed
  around in thread-local-storage to avoid threading through a parameter like
  `&mut Context` everywhere. An alternative would be to explicitly pass this
  parameter, but it might hinder trait implementations like `fmt::Display` and
  `FromStr`. Additionally, threading an extra parameter could perhaps become
  unwieldy over time.

* In addition to allowing definition of custom-derive forms, definition of
  custom procedural macros could also be allowed. They are similarly
  transformers from token streams to token streams, so the interface in this RFC
  would perhaps be appropriate. This addition, however, adds more surface area
  to this RFC and the macro 1.1 system which may not be necessary in the long
  run. It's currently understood that *only* custom derive is needed to move
  crates like serde and diesel onto stable Rust.

* Instead of having a global namespace of `#[derive]` modes which `rustc-macro`
  crates append to, we could at least require something along the lines of
  `#[derive(serde_macros::Deserialize)]`. This is unfortunately, however, still
  disconnected from what name resolution will actually be eventually and also
  deviates from what you actually may want, `#[derive(serde::Deserialize)]`, for
  example.

# Unresolved questions
[unresolved]: #unresolved-questions

* Is the interface between macros and the compiler actually general enough to
  be implemented differently one day?

* The intention of macros 1.1 is to be *as close as possible* to macros 2.0 in
  spirit and implementation, just without stabilizing vast quantities of
  features. In that sense, it is the intention that given a stable macros 1.1,
  we can layer on features backwards-compatibly to get to macros 2.0. Right now,
  though, the delta between what this RFC proposes and where we'd like to is
  very small, and can get get it down to actually zero?

* Eventually macro crates will want to be loaded both at compile time and
  runtime, and this means that Cargo will need to understand to compile these
  crates twice, once as `rustc-macro` and once as an rlib. Does Cargo have
  enough information to do this? Are the extensions needed here
  backwards-compatible?

* What sort of guarantees will be provided about the runtime environment for
  plugins? Are they sandboxed? Are they run in the same process?

* Should the name of this library be `rustc_macros`? The `rustc_` prefix
  normally means "private". Other alternatives are `macro` (make it a contextual
  keyword), `macros`, `proc_macro`.

* Should a `Context` or similar style argument be threaded through the APIs?
  Right now they sort of implicitly require one to be threaded through
  thread-local-storage.

* Should the APIs here be namespaced, perhaps with a `_1_1` suffix?

* To what extent can we preserve span information through heuristics?
  Should we adopt a slightly different API, for example one based on
  concatenation, to allow preserving spans?

