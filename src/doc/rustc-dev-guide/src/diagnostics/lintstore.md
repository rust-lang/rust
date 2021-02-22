# Lints

This page documents some of the machinery around lint registration and how we
run lints in the compiler.

The `LintStore` is the central piece of infrastructure, around which everything
rotates. It's not available during the early parts of compilation (i.e., before
TyCtxt) in most code, as we need to fill it in with all of the lints, which can only happen after
plugin registration.

## Lints vs. lint passes

There are two parts to the linting mechanism within the compiler: lints and lint passes.
Unfortunately, a lot of the documentation we have refers to both of these as just "lints."

First, we have the lint declarations themselves: this is where the name and default lint level and
other metadata come from. These are normally defined by way of the [`declare_lint!`] macro, which
boils down to a static with type `&rustc_session::lint::Lint`.

As of <!-- date: 2021-01 --> January 2021, we lint against direct declarations
without the use of the macro today (although this may change in the future, as
the macro is somewhat unwieldy to add new fields to, like all macros by
example).

Lint declarations don't carry any "state" - they are merely global identifers and descriptions of
lints. We assert at runtime that they are not registered twice (by lint name).

Lint passes are the meat of any lint. Notably, there is not a one-to-one relationship between
lints and lint passes; a lint might not have any lint pass that emits it, it could have many, or
just one -- the compiler doesn't track whether a pass is in any way associated with a particular
lint, and frequently lints are emitted as part of other work (e.g., type checking, etc.).

## Registration

### High-level overview

The lint store is created and all lints are registered during plugin registration, in
[`rustc_interface::register_plugins`]. There are three 'sources' of lint: the internal lints, plugin
lints, and `rustc_interface::Config` [`register_lints`]. All are registered here, in
`register_plugins`.

Once the registration is complete, we "freeze" the lint store by placing it in an `Lrc`. Later in
the driver, it's passed into the `GlobalCtxt` constructor where it lives in an immutable form from
then on.

Lints are registered via the [`LintStore::register_lint`] function. This should
happen just once for any lint, or an ICE will occur.

Lint passes are registered separately into one of the categories (pre-expansion,
early, late, late module). Passes are registered as a closure -- i.e., `impl
Fn() -> Box<dyn X>`, where `dyn X` is either an early or late lint pass trait
object. When we run the lint passes, we run the closure and then invoke the lint
pass methods, which take `&mut self` -- lint passes can keep track of state
internally.

#### Internal lints

Note, these include both rustc-internal lints, and the traditional lints, like, for example the dead
code lint.

These are primarily described in two places: `rustc_session::lint::builtin` and
`rustc_lint::builtin`. The first provides the definitions for the lints themselves,
and the latter provides the lint pass definitions (and implementations).

The internal lint registration happens in the [`rustc_lint::register_builtins`] function, along with
the [`rustc_lint::register_internals`] function. More generally, the LintStore "constructor"
function which is *the* way to get a `LintStore` in the compiler (you should not construct it
directly) is [`rustc_lint::new_lint_store`]; it calls the registration functions.

#### Plugin lints

This is one of the primary use cases remaining for plugins/drivers. Plugins are given access to the
mutable `LintStore` during registration to call any functions they need on the `LintStore`, just
like rustc code. Plugins are intended to declare lints with the `plugin` field set to true (e.g., by
way of the [`declare_tool_lint!`] macro), but this is purely for diagnostics and help text;
otherwise plugin lints are mostly just as first class as rustc builtin lints.

#### Driver lints

These are the lints provided by drivers via the `rustc_interface::Config` [`register_lints`] field,
which is a callback. Drivers should, if finding it already set, call the function currently set
within the callback they add. The best way for drivers to get access to this is by overriding the
`Callbacks::config` function which gives them direct access to the `Config` structure.

## Compiler lint passes are combined into one pass

Within the compiler, for performance reasons, we usually do not register dozens
of lint passes. Instead, we have a single lint pass of each variety
(e.g. `BuiltinCombinedModuleLateLintPass`) which will internally call all of the
individual lint passes; this is because then we get the benefits of static over
dynamic dispatch for each of the (often empty) trait methods.

Ideally, we'd not have to do this, since it certainly adds to the complexity of
understanding the code. However, with the current type-erased lint store
approach, it is beneficial to do so for performance reasons.

New lints being added likely want to join one of the existing declarations like
`late_lint_mod_passes` in `rustc_lint/src/lib.rs`, which would then
auto-propagate into the other.

[`LintStore::register_lint`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/struct.LintStore.html#method.register_lints
[`rustc_interface::register_plugins`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/passes/fn.register_plugins.html
[`rustc_lint::register_builtins`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/fn.register_builtins.html
[`rustc_lint::register_internals`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/fn.register_internals.html
[`rustc_lint::new_lint_store`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/fn.new_lint_store.html
[`declare_lint!`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/macro.declare_lint.html
[`declare_tool_lint!`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/macro.declare_tool_lint.html
[`register_lints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Config.html#structfield.register_lints
