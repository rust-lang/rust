## rust-analyzer.assist.emitMustUse {#assist.emitMustUse}

Default: `false`

Insert #[must_use] when generating `as_` methods for enum variants.


## rust-analyzer.assist.expressionFillDefault {#assist.expressionFillDefault}

Default: `"todo"`

Placeholder expression to use for missing expressions in assists.


## rust-analyzer.assist.preferSelf {#assist.preferSelf}

Default: `false`

Prefer to use `Self` over the type name when inserting a type (e.g. in "fill match arms" assist).


## rust-analyzer.assist.termSearch.borrowcheck {#assist.termSearch.borrowcheck}

Default: `true`

Enable borrow checking for term search code assists. If set to false, also there will be
more suggestions, but some of them may not borrow-check.


## rust-analyzer.assist.termSearch.fuel {#assist.termSearch.fuel}

Default: `1800`

Term search fuel in "units of work" for assists (Defaults to 1800).


## rust-analyzer.cachePriming.enable {#cachePriming.enable}

Default: `true`

Warm up caches on project load.


## rust-analyzer.cachePriming.numThreads {#cachePriming.numThreads}

Default: `"physical"`

How many worker threads to handle priming caches. The default `0` means to pick
automatically.


## rust-analyzer.cargo.allTargets {#cargo.allTargets}

Default: `true`

Pass `--all-targets` to cargo invocation.


## rust-analyzer.cargo.autoreload {#cargo.autoreload}

Default: `true`

Automatically refresh project info via `cargo metadata` on
`Cargo.toml` or `.cargo/config.toml` changes.


## rust-analyzer.cargo.buildScripts.enable {#cargo.buildScripts.enable}

Default: `true`

Run build scripts (`build.rs`) for more precise code analysis.


## rust-analyzer.cargo.buildScripts.invocationStrategy {#cargo.buildScripts.invocationStrategy}

Default: `"per_workspace"`

Specifies the invocation strategy to use when running the build scripts command.
If `per_workspace` is set, the command will be executed for each Rust workspace with the
workspace as the working directory.
If `once` is set, the command will be executed once with the opened project as the
working directory.
This config only has an effect when `#rust-analyzer.cargo.buildScripts.overrideCommand#`
is set.


## rust-analyzer.cargo.buildScripts.overrideCommand {#cargo.buildScripts.overrideCommand}

Default: `null`

Override the command rust-analyzer uses to run build scripts and
build procedural macros. The command is required to output json
and should therefore include `--message-format=json` or a similar
option.

If there are multiple linked projects/workspaces, this command is invoked for
each of them, with the working directory being the workspace root
(i.e., the folder containing the `Cargo.toml`). This can be overwritten
by changing `#rust-analyzer.cargo.buildScripts.invocationStrategy#`.

By default, a cargo invocation will be constructed for the configured
targets and features, with the following base command line:

```bash
cargo check --quiet --workspace --message-format=json --all-targets --keep-going
```

Note: The option must be specified as an array of command line arguments, with
the first argument being the name of the command to run.


## rust-analyzer.cargo.buildScripts.rebuildOnSave {#cargo.buildScripts.rebuildOnSave}

Default: `true`

Rerun proc-macros building/build-scripts running when proc-macro
or build-script sources change and are saved.


## rust-analyzer.cargo.buildScripts.useRustcWrapper {#cargo.buildScripts.useRustcWrapper}

Default: `true`

Use `RUSTC_WRAPPER=rust-analyzer` when running build scripts to
avoid checking unnecessary things.


## rust-analyzer.cargo.cfgs {#cargo.cfgs}

Default:
```json
[
  "debug_assertions",
  "miri"
]
```

List of cfg options to enable with the given values.

To enable a name without a value, use `"key"`.
To enable a name with a value, use `"key=value"`.
To disable, prefix the entry with a `!`.


## rust-analyzer.cargo.extraArgs {#cargo.extraArgs}

Default: `[]`

Extra arguments that are passed to every cargo invocation.


## rust-analyzer.cargo.extraEnv {#cargo.extraEnv}

Default: `{}`

Extra environment variables that will be set when running cargo, rustc
or other commands within the workspace. Useful for setting RUSTFLAGS.


## rust-analyzer.cargo.features {#cargo.features}

Default: `[]`

List of features to activate.

Set this to `"all"` to pass `--all-features` to cargo.


## rust-analyzer.cargo.noDefaultFeatures {#cargo.noDefaultFeatures}

Default: `false`

Whether to pass `--no-default-features` to cargo.


## rust-analyzer.cargo.noDeps {#cargo.noDeps}

Default: `false`

Whether to skip fetching dependencies. If set to "true", the analysis is performed
entirely offline, and Cargo metadata for dependencies is not fetched.


## rust-analyzer.cargo.sysroot {#cargo.sysroot}

Default: `"discover"`

Relative path to the sysroot, or "discover" to try to automatically find it via
"rustc --print sysroot".

Unsetting this disables sysroot loading.

This option does not take effect until rust-analyzer is restarted.


## rust-analyzer.cargo.sysrootSrc {#cargo.sysrootSrc}

Default: `null`

Relative path to the sysroot library sources. If left unset, this will default to
`{cargo.sysroot}/lib/rustlib/src/rust/library`.

This option does not take effect until rust-analyzer is restarted.


## rust-analyzer.cargo.target {#cargo.target}

Default: `null`

Compilation target override (target tuple).


## rust-analyzer.cargo.targetDir {#cargo.targetDir}

Default: `null`

Optional path to a rust-analyzer specific target directory.
This prevents rust-analyzer's `cargo check` and initial build-script and proc-macro
building from locking the `Cargo.lock` at the expense of duplicating build artifacts.

Set to `true` to use a subdirectory of the existing target directory or
set to a path relative to the workspace to use that path.


## rust-analyzer.cfg.setTest {#cfg.setTest}

Default: `true`

Set `cfg(test)` for local crates. Defaults to true.


## rust-analyzer.checkOnSave {#checkOnSave}

Default: `true`

Run the check command for diagnostics on save.


## rust-analyzer.check.allTargets {#check.allTargets}

Default: `null`

Check all targets and tests (`--all-targets`). Defaults to
`#rust-analyzer.cargo.allTargets#`.


## rust-analyzer.check.command {#check.command}

Default: `"check"`

Cargo command to use for `cargo check`.


## rust-analyzer.check.extraArgs {#check.extraArgs}

Default: `[]`

Extra arguments for `cargo check`.


## rust-analyzer.check.extraEnv {#check.extraEnv}

Default: `{}`

Extra environment variables that will be set when running `cargo check`.
Extends `#rust-analyzer.cargo.extraEnv#`.


## rust-analyzer.check.features {#check.features}

Default: `null`

List of features to activate. Defaults to
`#rust-analyzer.cargo.features#`.

Set to `"all"` to pass `--all-features` to Cargo.


## rust-analyzer.check.ignore {#check.ignore}

Default: `[]`

List of `cargo check` (or other command specified in `check.command`) diagnostics to ignore.

For example for `cargo check`: `dead_code`, `unused_imports`, `unused_variables`,...


## rust-analyzer.check.invocationStrategy {#check.invocationStrategy}

Default: `"per_workspace"`

Specifies the invocation strategy to use when running the check command.
If `per_workspace` is set, the command will be executed for each workspace.
If `once` is set, the command will be executed once.
This config only has an effect when `#rust-analyzer.check.overrideCommand#`
is set.


## rust-analyzer.check.noDefaultFeatures {#check.noDefaultFeatures}

Default: `null`

Whether to pass `--no-default-features` to Cargo. Defaults to
`#rust-analyzer.cargo.noDefaultFeatures#`.


## rust-analyzer.check.overrideCommand {#check.overrideCommand}

Default: `null`

Override the command rust-analyzer uses instead of `cargo check` for
diagnostics on save. The command is required to output json and
should therefore include `--message-format=json` or a similar option
(if your client supports the `colorDiagnosticOutput` experimental
capability, you can use `--message-format=json-diagnostic-rendered-ansi`).

If you're changing this because you're using some tool wrapping
Cargo, you might also want to change
`#rust-analyzer.cargo.buildScripts.overrideCommand#`.

If there are multiple linked projects/workspaces, this command is invoked for
each of them, with the working directory being the workspace root
(i.e., the folder containing the `Cargo.toml`). This can be overwritten
by changing `#rust-analyzer.check.invocationStrategy#`.

It supports two interpolation syntaxes, both mainly intended to be used with
[non-Cargo build systems](./non_cargo_based_projects.md):

- If `{saved_file}` is part of the command, rust-analyzer will pass
  the absolute path of the saved file to the provided command.
  (A previous version, `$saved_file`, also works.)
- If `{label}` is part of the command, rust-analyzer will pass the
  Cargo package ID, which can be used with `cargo check -p`, or a build label from
  `rust-project.json`. If `{label}` is included, rust-analyzer behaves much like
  [`"rust-analyzer.check.workspace": false`](#check.workspace).



An example command would be:

```bash
cargo check --workspace --message-format=json --all-targets
```

Note: The option must be specified as an array of command line arguments, with
the first argument being the name of the command to run.


## rust-analyzer.check.targets {#check.targets}

Default: `null`

Check for specific targets. Defaults to `#rust-analyzer.cargo.target#` if empty.

Can be a single target, e.g. `"x86_64-unknown-linux-gnu"` or a list of targets, e.g.
`["aarch64-apple-darwin", "x86_64-apple-darwin"]`.

Aliased as `"checkOnSave.targets"`.


## rust-analyzer.check.workspace {#check.workspace}

Default: `true`

Whether `--workspace` should be passed to `cargo check`.
If false, `-p <package>` will be passed instead if applicable. In case it is not, no
check will be performed.


## rust-analyzer.completion.addSemicolonToUnit {#completion.addSemicolonToUnit}

Default: `true`

Automatically add a semicolon when completing unit-returning functions.

In `match` arms it completes a comma instead.


## rust-analyzer.completion.autoAwait.enable {#completion.autoAwait.enable}

Default: `true`

Show method calls and field accesses completions with `await` prefixed to them when
completing on a future.


## rust-analyzer.completion.autoIter.enable {#completion.autoIter.enable}

Default: `true`

Show method call completions with `iter()` or `into_iter()` prefixed to them when
completing on a type that has them.


## rust-analyzer.completion.autoimport.enable {#completion.autoimport.enable}

Default: `true`

Show completions that automatically add imports when completed.

Note that your client must specify the `additionalTextEdits` LSP client capability to
truly have this feature enabled.


## rust-analyzer.completion.autoimport.exclude {#completion.autoimport.exclude}

Default:
```json
[
  {
    "path": "core::borrow::Borrow",
    "type": "methods"
  },
  {
    "path": "core::borrow::BorrowMut",
    "type": "methods"
  }
]
```

A list of full paths to items to exclude from auto-importing completions.

Traits in this list won't have their methods suggested in completions unless the trait
is in scope.

You can either specify a string path which defaults to type "always" or use the more
verbose form `{ "path": "path::to::item", type: "always" }`.

For traits the type "methods" can be used to only exclude the methods but not the trait
itself.

This setting also inherits `#rust-analyzer.completion.excludeTraits#`.


## rust-analyzer.completion.autoself.enable {#completion.autoself.enable}

Default: `true`

Show method calls and field access completions with `self` prefixed to them when
inside a method.


## rust-analyzer.completion.callable.snippets {#completion.callable.snippets}

Default: `"fill_arguments"`

Add parenthesis and argument snippets when completing function.


## rust-analyzer.completion.excludeTraits {#completion.excludeTraits}

Default: `[]`

A list of full paths to traits whose methods to exclude from completion.

Methods from these traits won't be completed, even if the trait is in scope. However,
they will still be suggested on expressions whose type is `dyn Trait`, `impl Trait` or
`T where T: Trait`.

Note that the trait themselves can still be completed.


## rust-analyzer.completion.fullFunctionSignatures.enable {#completion.fullFunctionSignatures.enable}

Default: `false`

Show full function / method signatures in completion docs.


## rust-analyzer.completion.hideDeprecated {#completion.hideDeprecated}

Default: `false`

Omit deprecated items from completions. By default they are marked as deprecated but not
hidden.


## rust-analyzer.completion.limit {#completion.limit}

Default: `null`

Maximum number of completions to return. If `None`, the limit is infinite.


## rust-analyzer.completion.postfix.enable {#completion.postfix.enable}

Default: `true`

Show postfix snippets like `dbg`, `if`, `not`, etc.


## rust-analyzer.completion.privateEditable.enable {#completion.privateEditable.enable}

Default: `false`

Show completions of private items and fields that are defined in the current workspace
even if they are not visible at the current position.


## rust-analyzer.completion.snippets.custom {#completion.snippets.custom}

Default:
```json
{
  "Ok": {
    "postfix": "ok",
    "body": "Ok(${receiver})",
    "description": "Wrap the expression in a `Result::Ok`",
    "scope": "expr"
  },
  "Box::pin": {
    "postfix": "pinbox",
    "body": "Box::pin(${receiver})",
    "requires": "std::boxed::Box",
    "description": "Put the expression into a pinned `Box`",
    "scope": "expr"
  },
  "Arc::new": {
    "postfix": "arc",
    "body": "Arc::new(${receiver})",
    "requires": "std::sync::Arc",
    "description": "Put the expression into an `Arc`",
    "scope": "expr"
  },
  "Some": {
    "postfix": "some",
    "body": "Some(${receiver})",
    "description": "Wrap the expression in an `Option::Some`",
    "scope": "expr"
  },
  "Err": {
    "postfix": "err",
    "body": "Err(${receiver})",
    "description": "Wrap the expression in a `Result::Err`",
    "scope": "expr"
  },
  "Rc::new": {
    "postfix": "rc",
    "body": "Rc::new(${receiver})",
    "requires": "std::rc::Rc",
    "description": "Put the expression into an `Rc`",
    "scope": "expr"
  }
}
```

Custom completion snippets.


## rust-analyzer.completion.termSearch.enable {#completion.termSearch.enable}

Default: `false`

Enable term search based snippets like `Some(foo.bar().baz())`.


## rust-analyzer.completion.termSearch.fuel {#completion.termSearch.fuel}

Default: `1000`

Term search fuel in "units of work" for autocompletion (Defaults to 1000).


## rust-analyzer.diagnostics.disabled {#diagnostics.disabled}

Default: `[]`

List of rust-analyzer diagnostics to disable.


## rust-analyzer.diagnostics.enable {#diagnostics.enable}

Default: `true`

Show native rust-analyzer diagnostics.


## rust-analyzer.diagnostics.experimental.enable {#diagnostics.experimental.enable}

Default: `false`

Show experimental rust-analyzer diagnostics that might have more false positives than
usual.


## rust-analyzer.diagnostics.remapPrefix {#diagnostics.remapPrefix}

Default: `{}`

Map of prefixes to be substituted when parsing diagnostic file paths. This should be the
reverse mapping of what is passed to `rustc` as `--remap-path-prefix`.


## rust-analyzer.diagnostics.styleLints.enable {#diagnostics.styleLints.enable}

Default: `false`

Run additional style lints.


## rust-analyzer.diagnostics.warningsAsHint {#diagnostics.warningsAsHint}

Default: `[]`

List of warnings that should be displayed with hint severity.

The warnings will be indicated by faded text or three dots in code and will not show up
in the `Problems Panel`.


## rust-analyzer.diagnostics.warningsAsInfo {#diagnostics.warningsAsInfo}

Default: `[]`

List of warnings that should be displayed with info severity.

The warnings will be indicated by a blue squiggly underline in code and a blue icon in
the `Problems Panel`.


## rust-analyzer.document.symbol.search.excludeLocals {#document.symbol.search.excludeLocals}

Default: `true`

Exclude all locals from document symbol search.


## rust-analyzer.files.exclude {#files.exclude}

Default: `[]`

List of files to ignore

These paths (file/directories) will be ignored by rust-analyzer. They are relative to
the workspace root, and globs are not supported. You may also need to add the folders to
Code's `files.watcherExclude`.


## rust-analyzer.files.watcher {#files.watcher}

Default: `"client"`

Controls file watching implementation.


## rust-analyzer.gotoImplementations.filterAdjacentDerives {#gotoImplementations.filterAdjacentDerives}

Default: `false`

If this is `true`, when "Goto Implementations" and in "Implementations" lens, are triggered on a `struct` or `enum` or `union`, we filter out trait implementations that originate from `derive`s above the type.


## rust-analyzer.highlightRelated.branchExitPoints.enable {#highlightRelated.branchExitPoints.enable}

Default: `true`

Highlight related return values while the cursor is on any `match`, `if`, or match arm
arrow (`=>`).


## rust-analyzer.highlightRelated.breakPoints.enable {#highlightRelated.breakPoints.enable}

Default: `true`

Highlight related references while the cursor is on `break`, `loop`, `while`, or `for`
keywords.


## rust-analyzer.highlightRelated.closureCaptures.enable {#highlightRelated.closureCaptures.enable}

Default: `true`

Highlight all captures of a closure while the cursor is on the `|` or move keyword of a closure.


## rust-analyzer.highlightRelated.exitPoints.enable {#highlightRelated.exitPoints.enable}

Default: `true`

Highlight all exit points while the cursor is on any `return`, `?`, `fn`, or return type
arrow (`->`).


## rust-analyzer.highlightRelated.references.enable {#highlightRelated.references.enable}

Default: `true`

Highlight related references while the cursor is on any identifier.


## rust-analyzer.highlightRelated.yieldPoints.enable {#highlightRelated.yieldPoints.enable}

Default: `true`

Highlight all break points for a loop or block context while the cursor is on any
`async` or `await` keywords.


## rust-analyzer.hover.actions.debug.enable {#hover.actions.debug.enable}

Default: `true`

Show `Debug` action. Only applies when `#rust-analyzer.hover.actions.enable#` is set.


## rust-analyzer.hover.actions.enable {#hover.actions.enable}

Default: `true`

Show HoverActions in Rust files.


## rust-analyzer.hover.actions.gotoTypeDef.enable {#hover.actions.gotoTypeDef.enable}

Default: `true`

Show `Go to Type Definition` action. Only applies when
`#rust-analyzer.hover.actions.enable#` is set.


## rust-analyzer.hover.actions.implementations.enable {#hover.actions.implementations.enable}

Default: `true`

Show `Implementations` action. Only applies when `#rust-analyzer.hover.actions.enable#`
is set.


## rust-analyzer.hover.actions.references.enable {#hover.actions.references.enable}

Default: `false`

Show `References` action. Only applies when `#rust-analyzer.hover.actions.enable#` is
set.


## rust-analyzer.hover.actions.run.enable {#hover.actions.run.enable}

Default: `true`

Show `Run` action. Only applies when `#rust-analyzer.hover.actions.enable#` is set.


## rust-analyzer.hover.actions.updateTest.enable {#hover.actions.updateTest.enable}

Default: `true`

Show `Update Test` action. Only applies when `#rust-analyzer.hover.actions.enable#` and
`#rust-analyzer.hover.actions.run.enable#` are set.


## rust-analyzer.hover.documentation.enable {#hover.documentation.enable}

Default: `true`

Show documentation on hover.


## rust-analyzer.hover.documentation.keywords.enable {#hover.documentation.keywords.enable}

Default: `true`

Show keyword hover popups. Only applies when
`#rust-analyzer.hover.documentation.enable#` is set.


## rust-analyzer.hover.dropGlue.enable {#hover.dropGlue.enable}

Default: `true`

Show drop glue information on hover.


## rust-analyzer.hover.links.enable {#hover.links.enable}

Default: `true`

Use markdown syntax for links on hover.


## rust-analyzer.hover.maxSubstitutionLength {#hover.maxSubstitutionLength}

Default: `20`

Show what types are used as generic arguments in calls etc. on hover, and limit the max
length to show such types, beyond which they will be shown with ellipsis.

This can take three values: `null` means "unlimited", the string `"hide"` means to not
show generic substitutions at all, and a number means to limit them to X characters.

The default is 20 characters.


## rust-analyzer.hover.memoryLayout.alignment {#hover.memoryLayout.alignment}

Default: `"hexadecimal"`

How to render the align information in a memory layout hover.


## rust-analyzer.hover.memoryLayout.enable {#hover.memoryLayout.enable}

Default: `true`

Show memory layout data on hover.


## rust-analyzer.hover.memoryLayout.niches {#hover.memoryLayout.niches}

Default: `false`

How to render the niche information in a memory layout hover.


## rust-analyzer.hover.memoryLayout.offset {#hover.memoryLayout.offset}

Default: `"hexadecimal"`

How to render the offset information in a memory layout hover.


## rust-analyzer.hover.memoryLayout.padding {#hover.memoryLayout.padding}

Default: `null`

How to render the padding information in a memory layout hover.


## rust-analyzer.hover.memoryLayout.size {#hover.memoryLayout.size}

Default: `"both"`

How to render the size information in a memory layout hover.


## rust-analyzer.hover.show.enumVariants {#hover.show.enumVariants}

Default: `5`

How many variants of an enum to display when hovering on. Show none if empty.


## rust-analyzer.hover.show.fields {#hover.show.fields}

Default: `5`

How many fields of a struct, variant or union to display when hovering on. Show none if
empty.


## rust-analyzer.hover.show.traitAssocItems {#hover.show.traitAssocItems}

Default: `null`

How many associated items of a trait to display when hovering a trait.


## rust-analyzer.imports.granularity.enforce {#imports.granularity.enforce}

Default: `false`

Enforce the import granularity setting for all files. If set to false rust-analyzer will
try to keep import styles consistent per file.


## rust-analyzer.imports.granularity.group {#imports.granularity.group}

Default: `"crate"`

How imports should be grouped into use statements.


## rust-analyzer.imports.group.enable {#imports.group.enable}

Default: `true`

Group inserted imports by the [following
order](https://rust-analyzer.github.io/book/features.html#auto-import). Groups are
separated by newlines.


## rust-analyzer.imports.merge.glob {#imports.merge.glob}

Default: `true`

Allow import insertion to merge new imports into single path glob imports like `use
std::fmt::*;`.


## rust-analyzer.imports.preferNoStd {#imports.preferNoStd}

Default: `false`

Prefer to unconditionally use imports of the core and alloc crate, over the std crate.


## rust-analyzer.imports.preferPrelude {#imports.preferPrelude}

Default: `false`

Prefer import paths containing a `prelude` module.


## rust-analyzer.imports.prefix {#imports.prefix}

Default: `"crate"`

The path structure for newly inserted paths to use.


## rust-analyzer.imports.prefixExternPrelude {#imports.prefixExternPrelude}

Default: `false`

Prefix external (including std, core) crate imports with `::`.

E.g. `use ::std::io::Read;`.


## rust-analyzer.inlayHints.bindingModeHints.enable {#inlayHints.bindingModeHints.enable}

Default: `false`

Show inlay type hints for binding modes.


## rust-analyzer.inlayHints.chainingHints.enable {#inlayHints.chainingHints.enable}

Default: `true`

Show inlay type hints for method chains.


## rust-analyzer.inlayHints.closingBraceHints.enable {#inlayHints.closingBraceHints.enable}

Default: `true`

Show inlay hints after a closing `}` to indicate what item it belongs to.


## rust-analyzer.inlayHints.closingBraceHints.minLines {#inlayHints.closingBraceHints.minLines}

Default: `25`

Minimum number of lines required before the `}` until the hint is shown (set to 0 or 1
to always show them).


## rust-analyzer.inlayHints.closureCaptureHints.enable {#inlayHints.closureCaptureHints.enable}

Default: `false`

Show inlay hints for closure captures.


## rust-analyzer.inlayHints.closureReturnTypeHints.enable {#inlayHints.closureReturnTypeHints.enable}

Default: `"never"`

Show inlay type hints for return types of closures.


## rust-analyzer.inlayHints.closureStyle {#inlayHints.closureStyle}

Default: `"impl_fn"`

Closure notation in type and chaining inlay hints.


## rust-analyzer.inlayHints.discriminantHints.enable {#inlayHints.discriminantHints.enable}

Default: `"never"`

Show enum variant discriminant hints.


## rust-analyzer.inlayHints.expressionAdjustmentHints.disableReborrows {#inlayHints.expressionAdjustmentHints.disableReborrows}

Default: `true`

Disable reborrows in expression adjustments inlay hints.

Reborrows are a pair of a builtin deref then borrow, i.e. `&*`. They are inserted by the compiler but are mostly useless to the programmer.

Note: if the deref is not builtin (an overloaded deref), or the borrow is `&raw const`/`&raw mut`, they are not removed.


## rust-analyzer.inlayHints.expressionAdjustmentHints.enable {#inlayHints.expressionAdjustmentHints.enable}

Default: `"never"`

Show inlay hints for type adjustments.


## rust-analyzer.inlayHints.expressionAdjustmentHints.hideOutsideUnsafe {#inlayHints.expressionAdjustmentHints.hideOutsideUnsafe}

Default: `false`

Hide inlay hints for type adjustments outside of `unsafe` blocks.


## rust-analyzer.inlayHints.expressionAdjustmentHints.mode {#inlayHints.expressionAdjustmentHints.mode}

Default: `"prefix"`

Show inlay hints as postfix ops (`.*` instead of `*`, etc).


## rust-analyzer.inlayHints.genericParameterHints.const.enable {#inlayHints.genericParameterHints.const.enable}

Default: `true`

Show const generic parameter name inlay hints.


## rust-analyzer.inlayHints.genericParameterHints.lifetime.enable {#inlayHints.genericParameterHints.lifetime.enable}

Default: `false`

Show generic lifetime parameter name inlay hints.


## rust-analyzer.inlayHints.genericParameterHints.type.enable {#inlayHints.genericParameterHints.type.enable}

Default: `false`

Show generic type parameter name inlay hints.


## rust-analyzer.inlayHints.implicitDrops.enable {#inlayHints.implicitDrops.enable}

Default: `false`

Show implicit drop hints.


## rust-analyzer.inlayHints.implicitSizedBoundHints.enable {#inlayHints.implicitSizedBoundHints.enable}

Default: `false`

Show inlay hints for the implied type parameter `Sized` bound.


## rust-analyzer.inlayHints.impliedDynTraitHints.enable {#inlayHints.impliedDynTraitHints.enable}

Default: `true`

Show inlay hints for the implied `dyn` keyword in trait object types.


## rust-analyzer.inlayHints.lifetimeElisionHints.enable {#inlayHints.lifetimeElisionHints.enable}

Default: `"never"`

Show inlay type hints for elided lifetimes in function signatures.


## rust-analyzer.inlayHints.lifetimeElisionHints.useParameterNames {#inlayHints.lifetimeElisionHints.useParameterNames}

Default: `false`

Prefer using parameter names as the name for elided lifetime hints if possible.


## rust-analyzer.inlayHints.maxLength {#inlayHints.maxLength}

Default: `25`

Maximum length for inlay hints. Set to null to have an unlimited length.

**Note:** This is mostly a hint, and we don't guarantee to strictly follow the limit.


## rust-analyzer.inlayHints.parameterHints.enable {#inlayHints.parameterHints.enable}

Default: `true`

Show function parameter name inlay hints at the call site.


## rust-analyzer.inlayHints.parameterHints.missingArguments.enable {#inlayHints.parameterHints.missingArguments.enable}

Default: `false`

Show parameter name inlay hints for missing arguments at the call site.


## rust-analyzer.inlayHints.rangeExclusiveHints.enable {#inlayHints.rangeExclusiveHints.enable}

Default: `false`

Show exclusive range inlay hints.


## rust-analyzer.inlayHints.reborrowHints.enable {#inlayHints.reborrowHints.enable}

Default: `"never"`

Show inlay hints for compiler inserted reborrows.

This setting is deprecated in favor of
#rust-analyzer.inlayHints.expressionAdjustmentHints.enable#.


## rust-analyzer.inlayHints.renderColons {#inlayHints.renderColons}

Default: `true`

Whether to render leading colons for type hints, and trailing colons for parameter hints.


## rust-analyzer.inlayHints.typeHints.enable {#inlayHints.typeHints.enable}

Default: `true`

Show inlay type hints for variables.


## rust-analyzer.inlayHints.typeHints.hideClosureInitialization {#inlayHints.typeHints.hideClosureInitialization}

Default: `false`

Hide inlay type hints for `let` statements that initialize to a closure.

Only applies to closures with blocks, same as
`#rust-analyzer.inlayHints.closureReturnTypeHints.enable#`.


## rust-analyzer.inlayHints.typeHints.hideClosureParameter {#inlayHints.typeHints.hideClosureParameter}

Default: `false`

Hide inlay parameter type hints for closures.


## rust-analyzer.inlayHints.typeHints.hideInferredTypes {#inlayHints.typeHints.hideInferredTypes}

Default: `false`

Hide inlay type hints for inferred types.


## rust-analyzer.inlayHints.typeHints.hideNamedConstructor {#inlayHints.typeHints.hideNamedConstructor}

Default: `false`

Hide inlay type hints for constructors.


## rust-analyzer.interpret.tests {#interpret.tests}

Default: `false`

Enable the experimental support for interpreting tests.


## rust-analyzer.joinLines.joinAssignments {#joinLines.joinAssignments}

Default: `true`

Join lines merges consecutive declaration and initialization of an assignment.


## rust-analyzer.joinLines.joinElseIf {#joinLines.joinElseIf}

Default: `true`

Join lines inserts else between consecutive ifs.


## rust-analyzer.joinLines.removeTrailingComma {#joinLines.removeTrailingComma}

Default: `true`

Join lines removes trailing commas.


## rust-analyzer.joinLines.unwrapTrivialBlock {#joinLines.unwrapTrivialBlock}

Default: `true`

Join lines unwraps trivial blocks.


## rust-analyzer.lens.debug.enable {#lens.debug.enable}

Default: `true`

Show `Debug` lens. Only applies when `#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.enable {#lens.enable}

Default: `true`

Show CodeLens in Rust files.


## rust-analyzer.lens.implementations.enable {#lens.implementations.enable}

Default: `true`

Show `Implementations` lens. Only applies when `#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.location {#lens.location}

Default: `"above_name"`

Where to render annotations.


## rust-analyzer.lens.references.adt.enable {#lens.references.adt.enable}

Default: `false`

Show `References` lens for Struct, Enum, and Union. Only applies when
`#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.references.enumVariant.enable {#lens.references.enumVariant.enable}

Default: `false`

Show `References` lens for Enum Variants. Only applies when
`#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.references.method.enable {#lens.references.method.enable}

Default: `false`

Show `Method References` lens. Only applies when `#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.references.trait.enable {#lens.references.trait.enable}

Default: `false`

Show `References` lens for Trait. Only applies when `#rust-analyzer.lens.enable#` is
set.


## rust-analyzer.lens.run.enable {#lens.run.enable}

Default: `true`

Show `Run` lens. Only applies when `#rust-analyzer.lens.enable#` is set.


## rust-analyzer.lens.updateTest.enable {#lens.updateTest.enable}

Default: `true`

Show `Update Test` lens. Only applies when `#rust-analyzer.lens.enable#` and
`#rust-analyzer.lens.run.enable#` are set.


## rust-analyzer.linkedProjects {#linkedProjects}

Default: `[]`

Disable project auto-discovery in favor of explicitly specified set of projects.

Elements must be paths pointing to `Cargo.toml`, `rust-project.json`, `.rs` files (which
will be treated as standalone files) or JSON objects in `rust-project.json` format.


## rust-analyzer.lru.capacity {#lru.capacity}

Default: `null`

Number of syntax trees rust-analyzer keeps in memory. Defaults to 128.


## rust-analyzer.lru.query.capacities {#lru.query.capacities}

Default: `{}`

The LRU capacity of the specified queries.


## rust-analyzer.notifications.cargoTomlNotFound {#notifications.cargoTomlNotFound}

Default: `true`

Show `can't find Cargo.toml` error message.


## rust-analyzer.numThreads {#numThreads}

Default: `null`

The number of worker threads in the main loop. The default `null` means to pick
automatically.


## rust-analyzer.procMacro.attributes.enable {#procMacro.attributes.enable}

Default: `true`

Expand attribute macros. Requires `#rust-analyzer.procMacro.enable#` to be set.


## rust-analyzer.procMacro.enable {#procMacro.enable}

Default: `true`

Enable support for procedural macros, implies `#rust-analyzer.cargo.buildScripts.enable#`.


## rust-analyzer.procMacro.ignored {#procMacro.ignored}

Default: `{}`

These proc-macros will be ignored when trying to expand them.

This config takes a map of crate names with the exported proc-macro names to ignore as values.


## rust-analyzer.procMacro.processes {#procMacro.processes}

Default: `1`

Number of proc-macro server processes to spawn.

Controls how many independent `proc-macro-srv` processes rust-analyzer
runs in parallel to handle macro expansion.


## rust-analyzer.procMacro.server {#procMacro.server}

Default: `null`

Internal config, path to proc-macro server executable.


## rust-analyzer.profiling.memoryProfile {#profiling.memoryProfile}

Default: `null`

The path where to save memory profiling output.

**Note:** Memory profiling is not enabled by default in rust-analyzer builds, you need to build
from source for it.


## rust-analyzer.references.excludeImports {#references.excludeImports}

Default: `false`

Exclude imports from find-all-references.


## rust-analyzer.references.excludeTests {#references.excludeTests}

Default: `false`

Exclude tests from find-all-references and call-hierarchy.


## rust-analyzer.rename.showConflicts {#rename.showConflicts}

Default: `true`

Whether to warn when a rename will cause conflicts (change the meaning of the code).


## rust-analyzer.runnables.bench.command {#runnables.bench.command}

Default: `"bench"`

Subcommand used for bench runnables instead of `bench`.


## rust-analyzer.runnables.bench.overrideCommand {#runnables.bench.overrideCommand}

Default: `null`

Override the command used for bench runnables.
The first element of the array should be the program to execute (for example, `cargo`).

Use the placeholders `${package}`, `${target_arg}`, `${target}`, `${test_name}` to dynamically
replace the package name, target option (such as `--bin` or `--example`), the target name and
the test name (name of test function or test mod path).


## rust-analyzer.runnables.command {#runnables.command}

Default: `null`

Command to be executed instead of 'cargo' for runnables.


## rust-analyzer.runnables.doctest.overrideCommand {#runnables.doctest.overrideCommand}

Default: `null`

Override the command used for bench runnables.
The first element of the array should be the program to execute (for example, `cargo`).

Use the placeholders `${package}`, `${target_arg}`, `${target}`, `${test_name}` to dynamically
replace the package name, target option (such as `--bin` or `--example`), the target name and
the test name (name of test function or test mod path).


## rust-analyzer.runnables.extraArgs {#runnables.extraArgs}

Default: `[]`

Additional arguments to be passed to cargo for runnables such as
tests or binaries. For example, it may be `--release`.


## rust-analyzer.runnables.extraTestBinaryArgs {#runnables.extraTestBinaryArgs}

Default:
```json
[
  "--nocapture"
]
```

Additional arguments to be passed through Cargo to launched tests, benchmarks, or
doc-tests.

Unless the launched target uses a
[custom test harness](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#the-harness-field),
they will end up being interpreted as options to
[`rustc`’s built-in test harness (“libtest”)](https://doc.rust-lang.org/rustc/tests/index.html#cli-arguments).


## rust-analyzer.runnables.test.command {#runnables.test.command}

Default: `"test"`

Subcommand used for test runnables instead of `test`.


## rust-analyzer.runnables.test.overrideCommand {#runnables.test.overrideCommand}

Default: `null`

Override the command used for test runnables.
The first element of the array should be the program to execute (for example, `cargo`).

Use the placeholders `${package}`, `${target_arg}`, `${target}`, `${test_name}` to dynamically
replace the package name, target option (such as `--bin` or `--example`), the target name and
the test name (name of test function or test mod path).


## rust-analyzer.rustc.source {#rustc.source}

Default: `null`

Path to the Cargo.toml of the rust compiler workspace, for usage in rustc_private
projects, or "discover" to try to automatically find it if the `rustc-dev` component
is installed.

Any project which uses rust-analyzer with the rustcPrivate
crates must set `[package.metadata.rust-analyzer] rustc_private=true` to use it.

This option does not take effect until rust-analyzer is restarted.


## rust-analyzer.rustfmt.extraArgs {#rustfmt.extraArgs}

Default: `[]`

Additional arguments to `rustfmt`.


## rust-analyzer.rustfmt.overrideCommand {#rustfmt.overrideCommand}

Default: `null`

Advanced option, fully override the command rust-analyzer uses for
formatting. This should be the equivalent of `rustfmt` here, and
not that of `cargo fmt`. The file contents will be passed on the
standard input and the formatted result will be read from the
standard output.

Note: The option must be specified as an array of command line arguments, with
the first argument being the name of the command to run.


## rust-analyzer.rustfmt.rangeFormatting.enable {#rustfmt.rangeFormatting.enable}

Default: `false`

Enables the use of rustfmt's unstable range formatting command for the
`textDocument/rangeFormatting` request. The rustfmt option is unstable and only
available on a nightly build.


## rust-analyzer.semanticHighlighting.comments.enable {#semanticHighlighting.comments.enable}

Default: `true`

Use semantic tokens for comments.

In some editors (e.g. vscode) semantic tokens override other highlighting grammars.
By disabling semantic tokens for comments, other grammars can be used to highlight
their contents.


## rust-analyzer.semanticHighlighting.doc.comment.inject.enable {#semanticHighlighting.doc.comment.inject.enable}

Default: `true`

Inject additional highlighting into doc comments.

When enabled, rust-analyzer will highlight rust source in doc comments as well as intra
doc links.


## rust-analyzer.semanticHighlighting.nonStandardTokens {#semanticHighlighting.nonStandardTokens}

Default: `true`

Emit non-standard tokens and modifiers

When enabled, rust-analyzer will emit tokens and modifiers that are not part of the
standard set of semantic tokens.


## rust-analyzer.semanticHighlighting.operator.enable {#semanticHighlighting.operator.enable}

Default: `true`

Use semantic tokens for operators.

When disabled, rust-analyzer will emit semantic tokens only for operator tokens when
they are tagged with modifiers.


## rust-analyzer.semanticHighlighting.operator.specialization.enable {#semanticHighlighting.operator.specialization.enable}

Default: `false`

Use specialized semantic tokens for operators.

When enabled, rust-analyzer will emit special token types for operator tokens instead
of the generic `operator` token type.


## rust-analyzer.semanticHighlighting.punctuation.enable {#semanticHighlighting.punctuation.enable}

Default: `false`

Use semantic tokens for punctuation.

When disabled, rust-analyzer will emit semantic tokens only for punctuation tokens when
they are tagged with modifiers or have a special role.


## rust-analyzer.semanticHighlighting.punctuation.separate.macro.bang {#semanticHighlighting.punctuation.separate.macro.bang}

Default: `false`

When enabled, rust-analyzer will emit a punctuation semantic token for the `!` of macro
calls.


## rust-analyzer.semanticHighlighting.punctuation.specialization.enable {#semanticHighlighting.punctuation.specialization.enable}

Default: `false`

Use specialized semantic tokens for punctuation.

When enabled, rust-analyzer will emit special token types for punctuation tokens instead
of the generic `punctuation` token type.


## rust-analyzer.semanticHighlighting.strings.enable {#semanticHighlighting.strings.enable}

Default: `true`

Use semantic tokens for strings.

In some editors (e.g. vscode) semantic tokens override other highlighting grammars.
By disabling semantic tokens for strings, other grammars can be used to highlight
their contents.


## rust-analyzer.signatureInfo.detail {#signatureInfo.detail}

Default: `"full"`

Show full signature of the callable. Only shows parameters if disabled.


## rust-analyzer.signatureInfo.documentation.enable {#signatureInfo.documentation.enable}

Default: `true`

Show documentation.


## rust-analyzer.typing.triggerChars {#typing.triggerChars}

Default: `"=."`

Specify the characters allowed to invoke special on typing triggers.

- typing `=` after `let` tries to smartly add `;` if `=` is followed by an existing
  expression
- typing `=` between two expressions adds `;` when in statement position
- typing `=` to turn an assignment into an equality comparison removes `;` when in
  expression position
- typing `.` in a chain method call auto-indents
- typing `{` or `(` in front of an expression inserts a closing `}` or `)` after the
  expression
- typing `{` in a use item adds a closing `}` in the right place
- typing `>` to complete a return type `->` will insert a whitespace after it
- typing `<` in a path or type position inserts a closing `>` after the path or type.


## rust-analyzer.vfs.extraIncludes {#vfs.extraIncludes}

Default: `[]`

Additional paths to include in the VFS. Generally for code that is
generated or otherwise managed by a build system outside of Cargo,
though Cargo might be the eventual consumer.


## rust-analyzer.workspace.discoverConfig {#workspace.discoverConfig}

Default: `null`

Configure a command that rust-analyzer can invoke to
obtain configuration.

This is an alternative to manually generating
`rust-project.json`: it enables rust-analyzer to generate
rust-project.json on the fly, and regenerate it when
switching or modifying projects.

This is an object with three fields:

* `command`: the shell command to invoke

* `filesToWatch`: which build system-specific files should
be watched to trigger regenerating the configuration

* `progressLabel`: the name of the command, used in
progress indicators in the IDE

Here's an example of a valid configuration:

```json
"rust-analyzer.workspace.discoverConfig": {
    "command": [
        "rust-project",
        "develop-json",
        "{arg}"
    ],
    "progressLabel": "buck2/rust-project",
    "filesToWatch": [
        "BUCK"
    ]
}
```

## Argument Substitutions

If `command` includes the argument `{arg}`, that argument will be substituted
with the JSON-serialized form of the following enum:

```norun
#[derive(PartialEq, Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum DiscoverArgument {
   Path(AbsPathBuf),
   Buildfile(AbsPathBuf),
}
```

rust-analyzer will use the path invocation to find and
generate a `rust-project.json` and therefore a
workspace. Example:


```norun
rust-project develop-json '{ "path": "myproject/src/main.rs" }'
```

rust-analyzer will use build file invocations to update an
existing workspace. Example:

Or with a build file and the configuration above:

```norun
rust-project develop-json '{ "buildfile": "myproject/BUCK" }'
```

As a reference for implementors, buck2's `rust-project`
will likely be useful:
<https://github.com/facebook/buck2/tree/main/integrations/rust-project>.

## Discover Command Output

**Warning**: This format is provisional and subject to change.

The discover command should output JSON objects, one per
line (JSONL format). These objects should correspond to
this Rust data type:

```norun
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind")]
#[serde(rename_all = "snake_case")]
enum DiscoverProjectData {
    Finished { buildfile: Utf8PathBuf, project: ProjectJsonData },
    Error { error: String, source: Option<String> },
    Progress { message: String },
}
```

For example, a progress event:

```json
{"kind":"progress","message":"generating rust-project.json"}
```

A finished event can look like this (expanded and
commented for readability):

```json
{
    // the internally-tagged representation of the enum.
    "kind": "finished",
    // the file used by a non-Cargo build system to define
    // a package or target.
    "buildfile": "rust-analyzer/BUCK",
    // the contents of a rust-project.json, elided for brevity
    "project": {
        "sysroot": "foo",
        "crates": []
    }
}
```

Only the finished event is required, but the other
variants are encouraged to give users more feedback about
progress or errors.


## rust-analyzer.workspace.symbol.search.excludeImports {#workspace.symbol.search.excludeImports}

Default: `false`

Exclude all imports from workspace symbol search.

In addition to regular imports (which are always excluded),
this option removes public imports (better known as re-exports)
and removes imports that rename the imported symbol.


## rust-analyzer.workspace.symbol.search.kind {#workspace.symbol.search.kind}

Default: `"only_types"`

Workspace symbol search kind.


## rust-analyzer.workspace.symbol.search.limit {#workspace.symbol.search.limit}

Default: `128`

Limits the number of items returned from a workspace symbol search (Defaults to 128).
Some clients like vs-code issue new searches on result filtering and don't require all results to be returned in the initial search.
Other clients requires all results upfront and might require a higher limit.


## rust-analyzer.workspace.symbol.search.scope {#workspace.symbol.search.scope}

Default: `"workspace"`

Workspace symbol search scope.


