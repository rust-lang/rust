- Start Date: 2014-07-16
- RFC PR #: [#169](https://github.com/rust-lang/rfcs/pull/169)
- Rust Issue #: https://github.com/rust-lang/rust/issues/16461

# Summary

Change the rebinding syntax from `use ID = PATH` to `use PATH as ID`,
so that paths all line up on the left side, and imported identifers
are all on the right side.  Also modify `extern crate` syntax
analogously, for consistency.

# Motivation

Currently, the view items at the start of a module look something like
this:

```rust
mod old_code {
  use a::b::c::d::www;
  use a::b::c::e::xxx;
  use yyy = a::b::yummy;
  use a::b::c::g::zzz;
}
```

This means that if you want to see what identifiers have been
imported, your eyes need to scan back and forth on both the left-hand
side (immediately beside the `use`) and the right-hand side (at the
end of each line).  In particular, note that `yummy` is *not* in scope
within the body of `old_code`

This RFC proposes changing the grammar of Rust so that the example
above would look like this:

```rust
mod new_code {
  use a::b::c::d::www;
  use a::b::c::e::xxx;
  use a::b::yummy as yyy;
  use a::b::c::g::zzz;
}
```

There are two benefits we can see by comparing `mod old_code` and `mod
new_code`:

 * As alluded to above, now all of the imported identfifiers are on
   the right-hand side of the block of view items.

 * Additionally, the left-hand side looks much more regular, since one
   sees the straight lines of `a::b::` characters all the way down,
   which makes the *actual* differences between the different paths
   more visually apparent.

# Detailed design

Currently, the grammar for use statements is something like:

```
  use_decl : "pub" ? "use" [ ident '=' path
                            | path_glob ] ;
```

Likewise, the grammar for extern crate declarations is something like:

```
  extern_crate_decl : "extern" "crate" ident [ '(' link_attrs ')' ] ? [ '=' string_lit ] ? ;
```

This RFC proposes changing the grammar for use statements to something like:

```
  use_decl : "pub" ? "use" [ path "as" ident
                            | path_glob ] ;
```

and the grammar for extern crate declarations to something like:

```
  extern_crate_decl : "extern" "crate" [ string_lit "as" ] ? ident [ '(' link_attrs ')' ] ? ;
```

Both `use` and `pub use` forms are changed to use `path as ident`
instead of `ident = path`.  The form `use path as ident` has the same
constraints and meaning that `use ident = path` has today.

Nothing about path globs is changed; the view items that use
`ident = path` are disjoint from the view items that use path globs,
and that continues to be the case under `path as ident`.

The old syntaxes
  `"use" ident '=' path`
and
  `"extern" "crate" ident '=' string_lit`
are removed (or at least deprecated).

# Drawbacks

* `pub use export = import_path` may be preferred over `pub use
  import_path as export` since people are used to seeing the name
  exported by a `pub` item on the left-hand side of an `=` sign.
  (See "Have distinct rebinding syntaxes for `use` and `pub use`"
  below.)

* The 'as' keyword is not currently used for any binding form in Rust.
  Adopting this RFC would change that precedent.
  (See "Change the signaling token" below.)

# Alternatives

## Keep things as they are

This just has the drawbacks outlined in the motivation: the left-hand
side of the view items are less regular, and one needs to scan both
the left- and right-hand sides to see all the imported identifiers.

## Change the signaling token

Go ahead with switch, so imported identifier is on the left-hand side,
but use a different token than `as` to signal a rebinding.

For example, we could use `@`, as an analogy with its use as a binding
operator in match expressions:

```rust
mod new_code {
  use a::b::c::d::www;
  use a::b::c::e::xxx;
  use a::b::yummy @ yyy;
  use a::b::c::g::zzz;
}
```
(I do not object to `path @ ident`, though I find it somehow more
"line-noisy" than `as` in this context.)

Or, we could use `=`:

```rust
mod new_code {
  use a::b::c::d::www;
  use a::b::c::e::xxx;
  use a::b::yummy = yyy;
  use a::b::c::g::zzz;
}
```
(I *do* object to `path = ident`, since typically when `=` is used to
bind, the identifier being bound occurs on the left-hand side.)

Or, we could use `:`, by (weak) analogy with struct pattern syntax:
```rust
mod new_code {
  use a::b::c::d::www;
  use a::b::c::e::xxx;
  use a::b::yummy : yyy;
  use a::b::c::g::zzz;
}
```
(I cannot figure out if this is genius or madness.  Probably madness,
especially if one is allowed to omit the whitespace around the `:`)

## Have distinct rebinding syntaxes for `use` and `pub use`

If people really like having `ident = path` for `pub use`, by the
reasoning presented above that people are used to seeing the name
exported by a `pub` item on the left-hand side of an `=` sign, then we
could support that by continuing to support `pub use ident = path`.

If we were to go down that route, I would prefer to have distinct
notions of the exported name and imported name, so that:

`pub use a = foo::bar;` would actually *import* `bar` (and `a` would
just be visible as an *export*), and then one could rebind for export
and import simultaneously, like so:
`pub use exported_bar = foo::bar as imported_bar;`

But really, is `pub use foo::bar as a` all that bad?

## Allow `extern crate ident as ident`

As written, this RFC allows for two variants of `extern_crate_decl`:

```rust
extern crate old_name;
extern crate "old_name" as new_name;
```

These are just analogous to the current options that use `=` instead of `as`.

However, the RFC comment dialogue suggested also allowing a renaming
form that does not use a string literal:

```rust
extern crate old_name as new_name;
```

I have no opinion on whether this should be added or not.  Arguably
this choice is orthgonal to the goals of this RFC (since, if this is a
good idea, it could just as well be implemented with the `=` syntax).
Perhaps it should just be filed as a separate RFC on its own.

# Unresolved questions

* In the revised `extern crate` form, is it best to put the
  `link_attrs` after the identifier, as written above?  Or would it be
  better for them to come after the `string_literal` when using the
  `extern crate string_literal as ident` form?
