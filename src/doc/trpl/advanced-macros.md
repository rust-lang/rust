% Advanced macros

This chapter picks up where the [introductory macro chapter](macros.html) left
off.

# Syntactic requirements

Even when Rust code contains un-expanded macros, it can be parsed as a full
[syntax tree][ast]. This property can be very useful for editors and other
tools that process code. It also has a few consequences for the design of
Rust's macro system.

[ast]: glossary.html#abstract-syntax-tree

One consequence is that Rust must determine, when it parses a macro invocation,
whether the macro stands in for

* zero or more items,
* zero or more methods,
* an expression,
* a statement, or
* a pattern.

A macro invocation within a block could stand for some items, or for an
expression / statement. Rust uses a simple rule to resolve this ambiguity. A
macro invocation that stands for items must be either

* delimited by curly braces, e.g. `foo! { ... }`, or
* terminated by a semicolon, e.g. `foo!(...);`

Another consequence of pre-expansion parsing is that the macro invocation must
consist of valid Rust tokens. Furthermore, parentheses, brackets, and braces
must be balanced within a macro invocation. For example, `foo!([)` is
forbidden. This allows Rust to know where the macro invocation ends.

More formally, the macro invocation body must be a sequence of *token trees*.
A token tree is defined recursively as either

* a sequence of token trees surrounded by matching `()`, `[]`, or `{}`, or
* any other single token.

Within a matcher, each metavariable has a *fragment specifier*, identifying
which syntactic form it matches.

* `ident`: an identifier. Examples: `x`; `foo`.
* `path`: a qualified name. Example: `T::SpecialA`.
* `expr`: an expression. Examples: `2 + 2`; `if true then { 1 } else { 2 }`; `f(42)`.
* `ty`: a type. Examples: `i32`; `Vec<(char, String)>`; `&T`.
* `pat`: a pattern. Examples: `Some(t)`; `(17, 'a')`; `_`.
* `stmt`: a single statement. Example: `let x = 3`.
* `block`: a brace-delimited sequence of statements. Example:
  `{ log(error, "hi"); return 12; }`.
* `item`: an [item][]. Examples: `fn foo() { }`; `struct Bar;`.
* `meta`: a "meta item", as found in attributes. Example: `cfg(target_os = "windows")`.
* `tt`: a single token tree.

There are additional rules regarding the next token after a metavariable:

* `expr` variables must be followed by one of: `=> , ;`
* `ty` and `path` variables must be followed by one of: `=> , : = > as`
* `pat` variables must be followed by one of: `=> , =`
* Other variables may be followed by any token.

These rules provide some flexibility for Rust's syntax to evolve without
breaking existing macros.

The macro system does not deal with parse ambiguity at all. For example, the
grammar `$($t:ty)* $e:expr` will always fail to parse, because the parser would
be forced to choose between parsing `$t` and parsing `$e`. Changing the
invocation syntax to put a distinctive token in front can solve the problem. In
this case, you can write `$(T $t:ty)* E $e:exp`.

[item]: ../reference.html#items

# Scoping and macro import/export

Macros are expanded at an early stage in compilation, before name resolution.
One downside is that scoping works differently for macros, compared to other
constructs in the language.

Definition and expansion of macros both happen in a single depth-first,
lexical-order traversal of a crate's source. So a macro defined at module scope
is visible to any subsequent code in the same module, which includes the body
of any subsequent child `mod` items.

A macro defined within the body of a single `fn`, or anywhere else not at
module scope, is visible only within that item.

If a module has the `macro_use` attribute, its macros are also visible in its
parent module after the child's `mod` item. If the parent also has `macro_use`
then the macros will be visible in the grandparent after the parent's `mod`
item, and so forth.

The `macro_use` attribute can also appear on `extern crate`. In this context
it controls which macros are loaded from the external crate, e.g.

```rust,ignore
#[macro_use(foo, bar)]
extern crate baz;
```

If the attribute is given simply as `#[macro_use]`, all macros are loaded. If
there is no `#[macro_use]` attribute then no macros are loaded. Only macros
defined with the `#[macro_export]` attribute may be loaded.

To load a crate's macros *without* linking it into the output, use `#[no_link]`
as well.

An example:

```rust
macro_rules! m1 { () => (()) }

// visible here: m1

mod foo {
    // visible here: m1

    #[macro_export]
    macro_rules! m2 { () => (()) }

    // visible here: m1, m2
}

// visible here: m1

macro_rules! m3 { () => (()) }

// visible here: m1, m3

#[macro_use]
mod bar {
    // visible here: m1, m3

    macro_rules! m4 { () => (()) }

    // visible here: m1, m3, m4
}

// visible here: m1, m3, m4
# fn main() { }
```

When this library is loaded with `#[macro_use] extern crate`, only `m2` will
be imported.

The Rust Reference has a [listing of macro-related
attributes](../reference.html#macro--and-plugin-related-attributes).

# The variable `$crate`

A further difficulty occurs when a macro is used in multiple crates. Say that
`mylib` defines

```rust
pub fn increment(x: u32) -> u32 {
    x + 1
}

#[macro_export]
macro_rules! inc_a {
    ($x:expr) => ( ::increment($x) )
}

#[macro_export]
macro_rules! inc_b {
    ($x:expr) => ( ::mylib::increment($x) )
}
# fn main() { }
```

`inc_a` only works within `mylib`, while `inc_b` only works outside the
library. Furthermore, `inc_b` will break if the user imports `mylib` under
another name.

Rust does not (yet) have a hygiene system for crate references, but it does
provide a simple workaround for this problem. Within a macro imported from a
crate named `foo`, the special macro variable `$crate` will expand to `::foo`.
By contrast, when a macro is defined and then used in the same crate, `$crate`
will expand to nothing. This means we can write

```rust
#[macro_export]
macro_rules! inc {
    ($x:expr) => ( $crate::increment($x) )
}
# fn main() { }
```

to define a single macro that works both inside and outside our library. The
function name will expand to either `::increment` or `::mylib::increment`.

To keep this system simple and correct, `#[macro_use] extern crate ...` may
only appear at the root of your crate, not inside `mod`. This ensures that
`$crate` is a single identifier.

# The deep end

The introductory chapter mentioned recursive macros, but it did not give the
full story. Recursive macros are useful for another reason: Each recursive
invocation gives you another opportunity to pattern-match the macro's
arguments.

As an extreme example, it is possible, though hardly advisable, to implement
the [Bitwise Cyclic Tag](http://esolangs.org/wiki/Bitwise_Cyclic_Tag) automaton
within Rust's macro system.

```rust
#![feature(trace_macros)]

macro_rules! bct {
    // cmd 0:  d ... => ...
    (0, $($ps:tt),* ; $_d:tt)
        => (bct!($($ps),*, 0 ; ));
    (0, $($ps:tt),* ; $_d:tt, $($ds:tt),*)
        => (bct!($($ps),*, 0 ; $($ds),*));

    // cmd 1p:  1 ... => 1 ... p
    (1, $p:tt, $($ps:tt),* ; 1)
        => (bct!($($ps),*, 1, $p ; 1, $p));
    (1, $p:tt, $($ps:tt),* ; 1, $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; 1, $($ds),*, $p));

    // cmd 1p:  0 ... => 0 ...
    (1, $p:tt, $($ps:tt),* ; $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; $($ds),*));

    // halt on empty data string
    ( $($ps:tt),* ; )
        => (());
}

fn main() {
    trace_macros!(true);
# /* just check the definition
    bct!(0, 0, 1, 1, 1 ; 1, 0, 1);
# */
}
```

Exercise: use macros to reduce duplication in the above definition of the
`bct!` macro.

# Procedural macros

If Rust's macro system can't do what you need, you may want to write a
[compiler plugin](plugins.html) instead. Compared to `macro_rules!`
macros, this is significantly more work, the interfaces are much less stable,
and bugs can be much harder to track down. In exchange you get the
flexibility of running arbitrary Rust code within the compiler. Syntax
extension plugins are sometimes called *procedural macros* for this reason.
