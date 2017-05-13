% Procedural Macros (and custom Derive)

As you've seen throughout the rest of the book, Rust provides a mechanism
called "derive" that lets you implement traits easily. For example,

```rust
#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
```

is a lot simpler than

```rust
struct Point {
    x: i32,
    y: i32,
}

use std::fmt;

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Point {{ x: {}, y: {} }}", self.x, self.y)
    }
}
```

Rust includes several traits that you can derive, but it also lets you define
your own. We can accomplish this task through a feature of Rust called
"procedural macros." Eventually, procedural macros will allow for all sorts of
advanced metaprogramming in Rust, but today, they're only for custom derive.

Let's build a very simple trait, and derive it with custom derive.

## Hello World

So the first thing we need to do is start a new crate for our project.

```bash
$ cargo new --bin hello-world
```

All we want is to be able to call `hello_world()` on a derived type. Something
like this:

```rust,ignore
#[derive(HelloWorld)]
struct Pancakes;

fn main() {
    Pancakes::hello_world();
}
```

With some kind of nice output, like `Hello, World! My name is Pancakes.`. 

Let's go ahead and write up what we think our macro will look like from a user
perspective. In `src/main.rs` we write:

```rust,ignore
#[macro_use]
extern crate hello_world_derive;

trait HelloWorld {
    fn hello_world();
}

#[derive(HelloWorld)]
struct FrenchToast;

#[derive(HelloWorld)]
struct Waffles;

fn main() {
    FrenchToast::hello_world();
    Waffles::hello_world();
}
```

Great. So now we just need to actually write the procedural macro. At the
moment, procedural macros need to be in their own crate. Eventually, this
restriction may be lifted, but for now, it's required. As such, there's a
convention; for a crate named `foo`, a custom derive procedural macro is called
`foo-derive`. Let's start a new crate called `hello-world-derive` inside our
`hello-world` project.

```bash
$ cargo new hello-world-derive
```

To make sure that our `hello-world` crate is able to find this new crate we've
created, we'll add it to our toml:

```toml
[dependencies]
hello-world-derive = { path = "hello-world-derive" }
```

As for our the source of our `hello-world-derive` crate, here's an example:

```rust,ignore
extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro_derive(HelloWorld)]
pub fn hello_world(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    let s = input.to_string();
    
    // Parse the string representation
    let ast = syn::parse_macro_input(&s).unwrap();

    // Build the impl
    let gen = impl_hello_world(&ast);
    
    // Return the generated impl
    gen.parse().unwrap()
}
```

So there is a lot going on here. We have introduced two new crates: [`syn`] and
[`quote`]. As you may have noticed, `input: TokenSteam` is immediately converted
to a `String`. This `String` is a string representation of the Rust code for which
we are deriving `HelloWorld` for. At the moment, the only thing you can do with a
`TokenStream` is convert it to a string. A richer API will exist in the future.

So what we really need is to be able to _parse_ Rust code into something
usable. This is where `syn` comes to play. `syn` is a crate for parsing Rust
code. The other crate we've introduced is `quote`. It's essentially the dual of
`syn` as it will make generating Rust code really easy. We could write this
stuff on our own, but it's much simpler to use these libraries. Writing a full
parser for Rust code is no simple task.

[`syn`]: https://crates.io/crates/syn
[`quote`]: https://crates.io/crates/quote

The comments seem to give us a pretty good idea of our overall strategy. We
are going to take a `String` of the Rust code for the type we are deriving, parse
it using `syn`, construct the implementation of `hello_world` (using `quote`),
then pass it back to Rust compiler.

One last note: you'll see some `unwrap()`s there. If you want to provide an
error for a procedural macro, then you should `panic!` with the error message.
In this case, we're keeping it as simple as possible.

Great, so let's write `impl_hello_world(&ast)`.

```rust,ignore
fn impl_hello_world(ast: &syn::MacroInput) -> quote::Tokens {
    let name = &ast.ident;
    quote! {
        impl HelloWorld for #name {
            fn hello_world() {
                println!("Hello, World! My name is {}", stringify!(#name));
            }
        }
    }
}
```

So this is where quotes comes in. The `ast` argument is a struct that gives us
a representation of our type (which can be either a `struct` or an `enum`).
Check out the [docs](https://docs.rs/syn/0.10.5/syn/struct.MacroInput.html),
there is some useful information there. We are able to get the name of the
type using `ast.ident`. The `quote!` macro let's us write up the Rust code
that we wish to return and convert it into `Tokens`. `quote!` let's us use some
really cool templating mechanics; we simply write `#name` and `quote!` will
replace it with the variable named `name`. You can even do some repetition
similar to regular macros work. You should check out the
[docs](https://docs.rs/quote) for a good introduction.

So I think that's it. Oh, well, we do need to add dependencies for `syn` and
`quote` in the `cargo.toml` for `hello-world-derive`.

```toml
[dependencies]
syn = "0.10.5"
quote = "0.3.10"
```

That should be it. Let's try to compile `hello-world`.

```bash
error: the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro` crate type
 --> hello-world-derive/src/lib.rs:8:3
  |
8 | #[proc_macro_derive(HelloWorld)]
  |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Oh, so it appears that we need to declare that our `hello-world-derive` crate is
a `proc-macro` crate type. How do we do this? Like this:

```toml
[lib]
proc-macro = true
```

Ok so now, let's compile `hello-world`. Executing `cargo run` now yields:

```bash
Hello, World! My name is FrenchToast
Hello, World! My name is Waffles
```

We've done it!
