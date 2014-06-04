- Start Date: 2014-05-23
- RFC PR #:
- Rust Issue #:

# Summary

Allow users to load custom lints into `rustc`, similar to loadable syntax extensions.

# Motivation

There are many possibilities for user-defined static checking:

* Enforcing correct usage of Servo's [JS-managed pointers](https://github.com/mozilla/servo/blob/master/src/components/script/dom/bindings/js.rs)
* kballard's use case: checking that `rust-lua` functions which call `longjmp` never have destructors on stack variables
* Enforcing a company or project style guide
* Detecting common misuses of a library, e.g. expensive or non-idiomatic constructs
* In cryptographic code, annotating which variables contain secrets and then forbidding their use in variable-time operations or memory addressing

Existing project-specific static checkers include:

* A [Clang plugin](https://tecnocode.co.uk/2013/12/09/clang-plugin-for-glib-and-gnome/) that detects misuse of GLib and GObject
* A [GCC plugin](https://gcc-python-plugin.readthedocs.org/en/latest/cpychecker.html) (written in Python!) that detects misuse of the CPython extension API
* [Sparse](https://sparse.wiki.kernel.org/index.php/Main_Page), which checks Linux kernel code for issues such as mixing up userspace and kernel pointers (often exploitable for privilege escalation)

We should make it easy to build such tools and integrate them with an existing Rust project.

# Detailed design

In `rustc::lint` (which today is `rustc::middle::lint`):

~~~ .rs
pub trait Lint {
    fn get_specs(&self) -> &'static [&'static LintSpec];

    fn check_item(&mut self, cx: &Context, it: &ast::Item) { }
    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) { }
    ...
}
~~~

To define a lint:

~~~ .rs
#![crate_id="lipogram"]
#![crate_type="dylib"]
#![feature(phase, plugin_registrar)]

extern crate syntax;

// Load rustc as a plugin to get the `declare_lint!` macro
#[phase(plugin, link)]
extern crate rustc;

use syntax::ast;
use syntax::parse::token;
use rustc::lint::{Lint, Context, Warn};
use rustc::plugin::Registry;

struct LetterE;

impl Lint for LetterE {
    // Defines `get_specs`; see below.
    declare_lint!("letter_e", "forbid use of the letter 'e'", Warn)

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let name = token::get_ident(it.ident);
        if name.get().contains_char('e') || name.get().contains_char('E') {
            cx.span_lint(it.span, "item name contains the letter 'e'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_lint(box LetterE as Box<Lint>);
}
~~~

To use the lint when compiling another crate:

~~~ .rs
#[phase(plugin)]
extern crate lipogram;

struct Foo;

#[allow(letter_e)]
struct Hello;
~~~

The macro `declare_lint!` is sugar to be used when defining a `Lint` impl which provides only one `LintSpec`.  This is the common case, but some lints will provide more than one.  For example `unstable`, `experimental`, and `deprecated` are implemented by the same chunk of code.  Here's how that looks:

~~~ .rs
struct Stability;

static unstable:     &'static LintSpec = &LintSpec { ... };
static experimental: &'static LintSpec = &LintSpec { ... };
static deprecated:   &'static LintSpec = &LintSpec { ... };

impl Lint for Stability {
    declare_lints!(unstable, experimental, deprecated)

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        if ... {
            cx.span_lint_for(unstable, it.span, "use of unstable item");
        }
    }
}
~~~

Internally, lints are identified by the address of a static `LintSpec`.  This has a number of benefits:

* The linker takes care of assigning unique IDs, even with dynamically loaded plugins.
* A typo writing a lint ID is usually a compiler error, unlike with string IDs.
* The ability to output a given lint is controlled by the usual visibility mechanism.  Lints defined within `rustc` use the same infrastructure and will simply export their `LintSpec`s if other parts of the compiler need to output those lints.
* IDs are small and easy to hash.
* It's easy to go from an ID to name, description, etc.

`cx.span_lint` is like `cx.span_lint_for` but implicitly uses the first `LintSpec` declared by this `Lint`, which avoids needing to name that `LintSpec` when there is only one.  (A lint with no `LintSpec`s shouldn't call `span_lint` or `span_lint_for` but can gather information for use by other code.)

User-defined lints are controlled through the usual mechanism of attributes and the `-A -W -D -F` flags to `rustc`.  User-defined lints will show up in `-W help` if a crate filename is also provided; otherwise we append a message suggesting to re-run with a crate filename.

# Drawbacks

This increases the amount of code in `rustc` to implement lints, although it makes each individual lint much easier to understand in isolation.

Loadable lints produce more coupling of user code to `rustc` internals (with no official stability guarantee, of course).

There's no scoping / namespacing of the lint name strings used by attributes and compiler flags.  Attempting to register a lint with a duplicate name is an error at registration time.

The use of `&'static` means that lint plugins can't dynamically generate the set of lints based on some external resource.

# Alternatives

We could provide a more generic mechanism for user-defined AST visitors.  This could support other use cases like code transformation.  But it would be harder to use, and harder to integrate with the lint infrastructure.

# Unresolved questions

Do we provide guarantees about visit order for a lint, or the order of multiple lints defined in the same crate?  Some lints may require multiple passes.

Since a `Lint` impl can provide multiple lints, should the trait have a different name?  I like the simplicity of `impl Lint for ...`, especially in the common case of one lint.

Should we enforce (while running lints) that each lint printed with `span_lint_for` was registered by the corresponding `Lint`?  Users who particularly care can already wrap lints in modules and use visibility to enforce this statically.

Should we separate registering a lint from initializing / constructing the value implementing `Lint`?  This would support a future where a single `rustc` invocation can compile multiple crates and needs to reset lint state.
