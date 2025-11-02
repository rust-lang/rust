# Editions

This chapter gives an overview of how Edition support works in rustc.
This assumes that you are familiar with what Editions are (see the [Edition Guide]).

[Edition Guide]: https://doc.rust-lang.org/edition-guide/

## Edition definition

The `--edition` CLI flag specifies the edition to use for a crate.
This can be accessed from [`Session::edition`].
There are convenience functions like [`Session::at_least_rust_2021`] for checking the crate's
edition, though you should be careful about whether you check the global session or the span, see
[Edition hygiene] below.

As an alternative to the `at_least_rust_20xx` convenience methods, the [`Edition`] type also
supports comparisons for doing range checks, such as `span.edition() >= Edition::Edition2021`.

[`Session::edition`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html#method.edition
[`Session::at_least_rust_2021`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html#method.at_least_rust_2021
[`Edition`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/edition/enum.Edition.html

### Adding a new edition

Adding a new edition mainly involves adding a variant to the [`Edition`] enum and then fixing
everything that is broken. See [#94461](https://github.com/rust-lang/rust/pull/94461) for an
example.

### Features and Edition stability

The [`Edition`] enum defines whether or not an edition is stable.
If it is not stable, then the `-Zunstable-options` CLI option must be passed to enable it.

When adding a new feature, there are two options you can choose for how to handle stability with a
future edition:

- Just check the edition of the span like `span.at_least_rust_20xx()` (see [Edition hygiene]) or the
  [`Session::edition`]. This will implicitly depend on the stability of the edition itself to
  indicate that your feature is available.
- Place your new behavior behind a [feature gate].

It may be sufficient to only check the current edition for relatively simple changes.
However, for larger language changes, you should consider creating a feature gate.
There are several benefits to using a feature gate:

- A feature gate makes it easier to work on and experiment with a new feature.
- It makes the intent clear when the `#![feature(…)]` attribute is used that your new feature is
  being enabled.
- It makes testing of editions easier so that features that are not yet complete do not interfere
  with testing of edition-specific features that are complete and ready.
- It decouples the feature from an edition, which makes it easier for the team to make a deliberate
  decision of whether or not a feature should be added to the next edition when the feature is
  ready.

When a feature is complete and ready, the feature gate can be removed (and the code should just
check the span or `Session` edition to determine if it is enabled).

There are a few different options for doing feature checks:

- For highly experimental features, that may or may not be involved in an edition, they can
  implement regular feature gates like `tcx.features().my_feature`, and ignore editions for the time
  being.

- For experimental features that *might* be involved in an edition, they should implement gates with
  `tcx.features().my_feature && span.at_least_rust_20xx()`.
  This requires the user to still specify `#![feature(my_feature)]`, to avoid disrupting testing of
  other edition features which are ready and have been accepted within the edition.

- For experimental features that have graduated to definitely be part of an edition,
  they should implement gates with `tcx.features().my_feature || span.at_least_rust_20xx()`,
  or just remove the feature check altogether and just check `span.at_least_rust_20xx()`.

If you need to do the feature gating in multiple places, consider placing the check in a single
function so that there will only be a single place to update. For example:

```rust,ignore
// An example from Edition 2021 disjoint closure captures.

fn enable_precise_capture(tcx: TyCtxt<'_>, span: Span) -> bool {
    tcx.features().capture_disjoint_fields || span.rust_2021()
}
```

See [Lints and stability](#lints-and-stability) below for more information about how lints handle
stability.

[feature gate]: ../feature-gates.md

## Edition parsing

For the most part, the lexer is edition-agnostic.
Within [`Lexer`], tokens can be modified based on edition-specific behavior.
For example, C-String literals like `c"foo"` are split into multiple tokens in editions before 2021.
This is also where things like reserved prefixes are handled for the 2021 edition.

Edition-specific parsing is relatively rare. One example is `async fn` which checks the span of the
token to determine if it is the 2015 edition, and emits an error in that case.
This can only be done if the syntax was already invalid.

If you need to do edition checking in the parser, you will normally want to look at the edition of
the token, see [Edition hygiene].
In some rare cases you may instead need to check the global edition from [`ParseSess::edition`].

Most edition-specific parsing behavior is handled with [migration lints] instead of in the parser.
This is appropriate when there is a *change* in syntax (as opposed to new syntax).
This allows the old syntax to continue to work on previous editions.
The lint then checks for the change in behavior.
On older editions, the lint pass should emit the migration lint to help with migrating to new
editions.
On newer editions, your code should emit a hard error with `emit_err` instead.
For example, the deprecated `start...end` pattern syntax emits the
[`ellipsis_inclusive_range_patterns`] lint on editions before 2021, and in 2021 is an hard error via
the `emit_err` method.

[`Lexer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/lexer/struct.Lexer.html
[`ParseSess::edition`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/struct.ParseSess.html#structfield.edition
[`ellipsis_inclusive_range_patterns`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/warn-by-default.html#ellipsis-inclusive-range-patterns

### Keywords

New keywords can be introduced across an edition boundary.
This is implemented by functions like [`Symbol::is_used_keyword_conditional`], which rely on the
ordering of how the keywords are defined.

When new keywords are introduced, the [`keyword_idents`] lint should be updated so that automatic
migrations can transition code that might be using the keyword as an identifier (see
[`KeywordIdents`]).
An alternative to consider is to implement the keyword as a weak keyword if the position it is used
is sufficient to distinguish it.

An additional option to consider is the `k#` prefix which was introduced in [RFC 3101].
This allows the use of a keyword in editions *before* the edition where the keyword is introduced.
This is currently not implemented.

[`Symbol::is_used_keyword_conditional`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html#method.is_used_keyword_conditional
[`keyword_idents`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/allowed-by-default.html#keyword-idents
[`KeywordIdents`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/builtin/struct.KeywordIdents.html
[RFC 3101]: https://rust-lang.github.io/rfcs/3101-reserved_prefixes.html

### Edition hygiene
[edition hygiene]: #edition-hygiene

Spans are marked with the edition of the crate that the span came from.
See [Macro hygiene] in the Edition Guide for a user-centric description of what this means.

You should normally use the edition from the token span instead of looking at the global `Session`
edition.
For example, use `span.edition().at_least_rust_2021()` instead of `sess.at_least_rust_2021()`.
This helps ensure that macros behave correctly when used across crates.

[Macro hygiene]: https://doc.rust-lang.org/nightly/edition-guide/editions/advanced-migrations.html#macro-hygiene

## Lints

Lints support a few different options for interacting with editions.
Lints can be *future incompatible edition migration lints*, which are used to support
[migrations][migration lints] to newer editions.
Alternatively, lints can be [edition-specific](#edition-specific-lints), where they change their
default level starting in a specific edition.

### Migration lints
[migration lints]: #migration-lints
[migration lint]: #migration-lints

*Migration lints* are used to migrate projects from one edition to the next.
They are implemented with a `MachineApplicable` [suggestion](../diagnostics.md#suggestions) which
will rewrite code so that it will **successfully compile in both the previous and the next
edition**.
For example, the [`keyword_idents`] lint will take identifiers that conflict with a new keyword to
use the raw identifier syntax to avoid the conflict (for example changing `async` to `r#async`).

Migration lints must be declared with the [`FutureIncompatibilityReason::EditionError`] or
[`FutureIncompatibilityReason::EditionSemanticsChange`] [future-incompatible
option](../diagnostics.md#future-incompatible-lints) in the lint declaration:

```rust,ignore
declare_lint! {
    pub KEYWORD_IDENTS,
    Allow,
    "detects edition keywords being used as an identifier",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
        reference: "issue #49716 <https://github.com/rust-lang/rust/issues/49716>",
    };
}
```

When declared like this, the lint is automatically added to the appropriate
`rust-20xx-compatibility` lint group.
When a user runs `cargo fix --edition`, cargo will pass the `--force-warn rust-20xx-compatibility`
flag to force all of these lints to appear during the edition migration.
Cargo also passes `--cap-lints=allow` so that no other lints interfere with the edition migration.

Make sure that the example code sets the correct edition. The example should illustrate the previous edition, and show what the migration warning would look like. For example, this lint for a 2024 migration shows an example in 2021:

```rust,ignore
declare_lint! {
    /// The `keyword_idents_2024` lint detects ...
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// #![warn(keyword_idents_2024)]
    /// fn gen() {}
    /// ```
    ///
    /// {{produces}}
}
```

Migration lints can be either `Allow` or `Warn` by default.
If it is `Allow`, users usually won't see this warning unless they are doing an edition migration
manually or there is a problem during the migration.
Most migration lints are `Allow`.

If it is `Warn` by default, users on all editions will see this warning.
Only use `Warn` if you think it is important for everyone to be aware of the change, and to
encourage people to update their code on all editions.
Beware that new warn-by-default lint that hit many projects can be very disruptive and frustrating
for users.
You may consider switching an `Allow` to `Warn` several years after the edition stabilizes.
This will only show up for the relatively small number of stragglers who have not updated to the new
edition.

[`keyword_idents`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/allowed-by-default.html#keyword-idents
[`FutureIncompatibilityReason::EditionError`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint_defs/enum.FutureIncompatibilityReason.html#variant.EditionError
[`FutureIncompatibilityReason::EditionSemanticsChange`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint_defs/enum.FutureIncompatibilityReason.html#variant.EditionSemanticsChange

### Edition-specific lints

Lints can be marked so that they have a different level starting in a specific edition.
In the lint declaration, use the `@edition` marker:

```rust,ignore
declare_lint! {
    pub SOME_LINT_NAME,
    Allow,
    "my lint description",
    @edition Edition2024 => Warn;
}
```

Here, `SOME_LINT_NAME` defaults to `Allow` on all editions before 2024, and then becomes `Warn`
afterwards.

This should generally be used sparingly, as there are other options:

- Small impact stylistic changes unrelated to an edition can just make the lint `Warn` on all
  editions. If you want people to adopt a different way to write things, then go ahead and commit to
  having it show up for all projects.

  Beware that if a new warn-by-default lint hits many projects, it can be very disruptive and
  frustrating for users.

- Change the new style to be a hard error in the new edition, and use a [migration lint] to
  automatically convert projects to the new style. For example,
  [`ellipsis_inclusive_range_patterns`] is a hard error in 2021, and warns in all previous editions.

  Beware that these cannot be added after the edition stabilizes.

- Migration lints can also change over time.
  For example, the migration lint can start out as `Allow` by default.
  For people performing the migration, they will automatically get updated to the new code.
  Then, after some years, the lint can be made to `Warn` in previous editions.

  For example [`anonymous_parameters`] was a 2018 Edition migration lint (and a hard-error in 2018)
  that was `Allow` by default in previous editions.
  Then, three years later, it was changed to `Warn` for all previous editions, so that all users got
  a warning that the style was being phased out.
  If this was a warning from the start, it would have impacted many projects and be very disruptive.
  By making it part of the edition, most users eventually updated to the new edition and were
  handled by the migration.
  Switching to `Warn` only impacted a few stragglers who did not update.

[`ellipsis_inclusive_range_patterns`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/warn-by-default.html#ellipsis-inclusive-range-patterns
[`anonymous_parameters`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/warn-by-default.html#anonymous-parameters

### Lints and stability

Lints can be marked as being unstable, which can be helpful when developing a new edition feature,
and you want to test out a migration lint.
The feature gate can be specified in the lint's declaration like this:

```rust,ignore
declare_lint! {
    pub SOME_LINT_NAME,
    Allow,
    "my cool lint",
    @feature_gate = sym::my_feature_name;
}
```

Then, the lint will only fire if the user has the appropriate `#![feature(my_feature_name)]`.
Just beware that when it comes time to do crater runs testing the migration that the feature gate
will need to be removed.

Alternatively, you can implement an allow-by-default [migration lint] for an upcoming unstable
edition without a feature gate.
Although users may technically be able to enable the lint before the edition is stabilized, most
will not notice the new lint exists, and it should not disrupt anything or cause any breakage.

### Idiom lints

In the 2018 edition, there was a concept of "idiom lints" under the `rust-2018-idioms` lint group.
The concept was to have new idiomatic styles under a different lint group separate from the forced
migrations under the `rust-2018-compatibility` lint group, giving some flexibility as to how people
opt-in to certain edition changes.

Overall this approach did not seem to work very well,
and it is unlikely that we will use the idiom groups in the future.

## Standard library changes

### Preludes

Each edition comes with a specific prelude of the standard library.
These are implemented as regular modules in [`core::prelude`] and [`std::prelude`].
New items can be added to the prelude, just beware that this can conflict with user's pre-existing
code.
Usually a [migration lint] should be used to migrate existing code to avoid the conflict.
For example, [`rust_2021_prelude_collisions`] is used to handle the collisions with the new traits
in 2021.

[`core::prelude`]: https://doc.rust-lang.org/core/prelude/index.html
[`std::prelude`]: https://doc.rust-lang.org/std/prelude/index.html
[`rust_2021_prelude_collisions`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/allowed-by-default.html#rust-2021-prelude-collisions

### Customized language behavior

Usually it is not possible to make breaking changes to the standard library.
In some rare cases, the teams may decide that the behavior change is important enough to break this
rule.
The downside is that this requires special handling in the compiler to be able to distinguish when
the old and new signatures or behaviors should be used.

One example is the change in method resolution for [`into_iter()` of arrays][into-iter].
This was implemented with the `#[rustc_skip_array_during_method_dispatch]` attribute on the
`IntoIterator` trait which then tells the compiler to consider an alternate trait resolution choice
based on the edition.

Another example is the [`panic!` macro changes][panic-macro].
This required defining multiple panic macros, and having the built-in panic macro implementation
determine the appropriate way to expand it.
This also included the [`non_fmt_panics`] [migration lint] to adjust old code to the new form, which
required the `rustc_diagnostic_item` attribute to detect the usage of the panic macro.

In general it is recommended to avoid these special cases except for very high value situations.

[into-iter]: https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html
[panic-macro]: https://doc.rust-lang.org/nightly/edition-guide/rust-2021/panic-macro-consistency.html
[`non_fmt_panics`]: https://doc.rust-lang.org/nightly/rustc/lints/listing/warn-by-default.html#non-fmt-panics

### Migrating the standard library edition

Updating the edition of the standard library itself roughly involves the following process:

- Wait until the newly stabilized edition has reached beta and the bootstrap compiler has been updated.
- Apply migration lints. This can be an involved process since some code is in external submodules[^std-submodules], and the standard library makes heavy use of conditional compilation. Also, running `cargo fix --edition` can be impractical on the standard library itself. One approach is to individually add `#![warn(...)]` at the top of each crate for each lint, run `./x check library`, apply the migrations, remove the `#![warn(...)]` and commit each migration separately. You'll likely need to run `./x check` with `--target` for many different targets to get full coverage (otherwise you'll likely spend days or weeks getting CI to pass)[^ed-docker]. See also the [advanced migration guide] for more tips.
    - Apply migrations to [`backtrace-rs`]. [Example for 2024](https://github.com/rust-lang/backtrace-rs/pull/700). Note that this doesn't update the edition of the crate itself because that is published independently on crates.io, and that would otherwise restrict the minimum Rust version. Consider adding some `#![deny()]` attributes to avoid regressions until its edition gets updated.
    - Apply migrations to [`stdarch`], and update its edition, and formatting. [Example for 2024](https://github.com/rust-lang/stdarch/pull/1710).
    - Post PRs to update the backtrace and stdarch submodules, and wait for those to land.
    - Apply migration lints to the standard library crates, and update their edition. I recommend working one crate at a time starting with `core`. [Example for 2024](https://github.com/rust-lang/rust/pull/138162).

[^std-submodules]: This will hopefully change in the future to pull these submodules into `rust-lang/rust`.
[^ed-docker]: You'll also likely need to do a lot of testing for different targets, and this is where [docker testing](../tests/docker.md) comes in handy.

[advanced migration guide]: https://doc.rust-lang.org/nightly/edition-guide/editions/advanced-migrations.html
[`backtrace-rs`]: https://github.com/rust-lang/backtrace-rs/
[`stdarch`]: https://github.com/rust-lang/stdarch/

## Stabilizing an edition

After the edition team has given the go-ahead, the process for stabilizing an edition is roughly:

- Update [`LATEST_STABLE_EDITION`].
- Update [`Edition::is_stable`].
- Hunt and find any document that refers to edition by number, and update it:
    - [`--edition` flag](https://github.com/rust-lang/rust/blob/master/src/doc/rustc/src/command-line-arguments.md#--edition-specify-the-edition-to-use)
    - [Rustdoc attributes](https://github.com/rust-lang/rust/blob/master/src/doc/rustdoc/src/write-documentation/documentation-tests.md#attributes)
- Clean up any tests that use the `//@ edition` header to remove the `-Zunstable-options` flag to ensure they are indeed stable. Note: Ideally this should be automated, see [#133582].
- Bless any tests that change.
- Update `lint-docs` to default to the new edition.

See [example for 2024](https://github.com/rust-lang/rust/pull/133349).

[`LATEST_STABLE_EDITION`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/edition/constant.LATEST_STABLE_EDITION.html
[`Edition::is_stable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/edition/enum.Edition.html#method.is_stable
[#133582]: https://github.com/rust-lang/rust/issues/133582
