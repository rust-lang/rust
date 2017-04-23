# `proc_macro`

The tracking issue for this feature is: [#38356]

[#38356]: https://github.com/rust-lang/rust/issues/38356

------------------------

This feature flag guards the new procedural macro features as laid out by [RFC 1566], which alongside the now-stable 
[custom derives], provide stabilizable alternatives to the compiler plugin API (which requires the use of 
perma-unstable internal APIs) for programmatically modifying Rust code at compile-time.

The two new procedural macro kinds are:
 
* Function-like procedural macros which are invoked like regular declarative macros, and:

* Attribute-like procedural macros which can be applied to any item which built-in attributes can
be applied to, and which can take arguments in their invocation as well.

Additionally, this feature flag implicitly enables the [`use_extern_macros`](language-features/use-extern-macros.html) feature,
which allows macros to be imported like any other item with `use` statements, as compared to 
applying `#[macro_use]` to an `extern crate` declaration. It is important to note that procedural macros may
**only** be imported in this manner, and will throw an error otherwise.

You **must** declare the `proc_macro` feature in both the crate declaring these new procedural macro kinds as well as 
in any crates that use them.

### Common Concepts

As with custom derives, procedural macros may only be declared in crates of the `proc-macro` type, and must be public
functions. No other public items may be declared in `proc-macro` crates, but private items are fine.

To declare your crate as a `proc-macro` crate, simply add:

```toml
[lib]
proc-macro = true
```

to your `Cargo.toml`. 

Unlike custom derives, however, the name of the function implementing the procedural macro is used directly as the 
procedural macro's name, so choose carefully.

Additionally, both new kinds of procedural macros return a `TokenStream` which *wholly* replaces the original 
invocation and its input.

#### Importing

As referenced above, the new procedural macros are not meant to be imported via `#[macro_use]` and will throw an 
error if they are. Instead, they are meant to be imported like any other item in Rust, with `use` statements:

```rust,ignore
#![feature(proc_macro)]

// Where `my_proc_macros` is some crate of type `proc_macro`
extern crate my_proc_macros;

// And declares a `#[proc_macro] pub fn my_bang_macro()` at its root.
use my_proc_macros::my_bang_macro;

fn main() {
    println!("{}", my_bang_macro!());
}
```

#### Error Reporting

Any panics in a procedural macro implementation will be caught by the compiler and turned into an error message pointing 
to the problematic invocation. Thus, it is important to make your panic messages as informative as possible: use 
`Option::expect` instead of `Option::unwrap` and `Result::expect` instead of `Result::unwrap`, and inform the user of 
the error condition as unambiguously as you can.
 
#### `TokenStream`

The `proc_macro::TokenStream` type is hardcoded into the signatures of procedural macro functions for both input and 
output. It is a wrapper around the compiler's internal representation for a given chunk of Rust code.

### Function-like Procedural Macros

These are procedural macros that are invoked like regular declarative macros. They are declared as public functions in 
crates of the `proc_macro` type and using the `#[proc_macro]` attribute. The name of the declared function becomes the 
name of the macro as it is to be imported and used. The function must be of the kind `fn(TokenStream) -> TokenStream` 
where the sole argument is the input to the macro and the return type is the macro's output.

This kind of macro can expand to anything that is valid for the context it is invoked in, including expressions and
statements, as well as items.

**Note**: invocations of this kind of macro require a wrapping `[]`, `{}` or `()` like regular macros, but these do not 
appear in the input, only the tokens between them. The tokens between the braces do not need to be valid Rust syntax.

<span class="filename">my_macro_crate/src/lib.rs</span>

```rust,ignore
#![feature(proc_macro)]

// This is always necessary to get the `TokenStream` typedef.
extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn say_hello(_input: TokenStream) -> TokenStream {
    // This macro will accept any input because it ignores it. 
    // To enforce correctness in macros which don't take input,
    // you may want to add `assert!(_input.to_string().is_empty());`.
    "println!(\"Hello, world!\")".parse().unwrap()
}
```

<span class="filename">my_macro_user/Cargo.toml</span>

```toml
[dependencies]
my_macro_crate = { path = "<relative path to my_macro_crate>" }
```

<span class="filename">my_macro_user/src/lib.rs</span>

```rust,ignore
#![feature(proc_macro)]

extern crate my_macro_crate;

use my_macro_crate::say_hello;

fn main() {
    say_hello!();
}
```

As expected, this prints `Hello, world!`.

### Attribute-like Procedural Macros

These are arguably the most powerful flavor of procedural macro as they can be applied anywhere attributes are allowed. 

They are declared as public functions in crates of the `proc-macro` type, using the `#[proc_macro_attribute]` attribute. 
The name of the function becomes the name of the attribute as it is to be imported and used. The function must be of the 
kind `fn(TokenStream, TokenStream) -> TokenStream` where:

The first argument represents any metadata for the attribute (see [the reference chapter on attributes][refr-attr]). 
Only the metadata itself will appear in this argument, for example:
 
 * `#[my_macro]` will get an empty string.
 * `#[my_macro = "string"]` will get `= "string"`.
 * `#[my_macro(ident)]` will get `(ident)`.
 * etc.
 
The second argument is the item that the attribute is applied to. It can be a function, a type definition, 
an impl block, an `extern` block, or a module—attribute invocations can take the inner form (`#![my_attr]`) 
or outer form (`#[my_attr]`).

The return type is the output of the macro which *wholly* replaces the item it was applied to. Thus, if your intention
is to merely modify an item, it *must* be copied to the output. The output must be an item; expressions, statements
and bare blocks are not allowed.

There is no restriction on how many items an attribute-like procedural macro can emit as long as they are valid in 
the given context.

<span class="filename">my_macro_crate/src/lib.rs</span>

```rust,ignore
#![feature(proc_macro)]

extern crate proc_macro;

use proc_macro::TokenStream;

/// Adds a `/// ### Panics` docstring to the end of the input's documentation
///
/// Does not assert that its receiver is a function or method.
#[proc_macro_attribute]
pub fn panics_note(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = args.to_string();
    let mut input = input.to_string();

    assert!(args.starts_with("= \""), "`#[panics_note]` requires an argument of the form \
                                       `#[panics_note = \"panic note here\"]`");

    // Get just the bare note string
    let panics_note = args.trim_matches(&['=', ' ', '"'][..]);

    // The input will include all docstrings regardless of where the attribute is placed,
    // so we need to find the last index before the start of the item
    let insert_idx = idx_after_last_docstring(&input);

    // And insert our `### Panics` note there so it always appears at the end of an item's docs
    input.insert_str(insert_idx, &format!("/// # Panics \n/// {}\n", panics_note));

    input.parse().unwrap()
}

// `proc-macro` crates can contain any kind of private item still
fn idx_after_last_docstring(input: &str) -> usize {
    // Skip docstring lines to find the start of the item proper
    input.lines().skip_while(|line| line.trim_left().starts_with("///")).next()
        // Find the index of the first non-docstring line in the input
        // Note: assumes this exact line is unique in the input
        .and_then(|line_after| input.find(line_after))
        // No docstrings in the input
        .unwrap_or(0)
}
```

<span class="filename">my_macro_user/Cargo.toml</span>

```toml
[dependencies]
my_macro_crate = { path = "<relative path to my_macro_crate>" }
```

<span class="filename">my_macro_user/src/lib.rs</span>

```rust,ignore
#![feature(proc_macro)]

extern crate my_macro_crate;

use my_macro_crate::panics_note;

/// Do the `foo` thing.
#[panics_note = "Always."]
pub fn foo() {
    panic!()
}
```

Then the rendered documentation for `pub fn foo` will look like this:

> `pub fn foo()`
> 
> ----
> Do the `foo` thing.
> # Panics
> Always.

[RFC 1566]: https://github.com/rust-lang/rfcs/blob/master/text/1566-proc-macros.md
[custom derives]: https://doc.rust-lang.org/book/procedural-macros.html
[rust-lang/rust#41430]: https://github.com/rust-lang/rust/issues/41430
[refr-attr]: https://doc.rust-lang.org/reference/attributes.html
