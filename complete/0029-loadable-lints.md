- Start Date: 2014-05-23
- RFC PR: [rust-lang/rfcs#89](https://github.com/rust-lang/rfcs/pull/89)
- Rust Issue: [rust-lang/rust#14067](https://github.com/rust-lang/rust/issues/14067)

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
pub struct Lint {
    /// An identifier for the lint, written with underscores,
    /// e.g. "unused_imports".
    pub name: &'static str,

    /// Default level for the lint.
    pub default_level: Level,

    /// Description of the lint or the issue it detects,
    /// e.g. "imports that are never used"
    pub desc: &'static str,
}

#[macro_export]
macro_rules! declare_lint ( ($name:ident, $level:ident, $desc:expr) => (
    static $name: &'static ::rustc::lint::Lint
        = &::rustc::lint::Lint {
            name: stringify!($name),
            default_level: ::rustc::lint::$level,
            desc: $desc,
        };
))

pub type LintArray = &'static [&'static Lint];

#[macro_export]
macro_rules! lint_array ( ($( $lint:expr ),*) => (
    {
        static array: LintArray = &[ $( $lint ),* ];
        array
    }
))

pub trait LintPass {
    fn get_lints(&self) -> LintArray;

    fn check_item(&mut self, cx: &Context, it: &ast::Item) { }
    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) { }
    ...
}

pub type LintPassObject = Box<LintPass: 'static>;
~~~

To define a lint:

~~~ .rs
#![crate_id="lipogram"]
#![crate_type="dylib"]
#![feature(phase, plugin_registrar)]

extern crate syntax;

// Load rustc as a plugin to get macros
#[phase(plugin, link)]
extern crate rustc;

use syntax::ast;
use syntax::parse::token;
use rustc::lint::{Context, LintPass, LintPassObject, LintArray};
use rustc::plugin::Registry;

declare_lint!(letter_e, Warn, "forbid use of the letter 'e'")

struct Lipogram;

impl LintPass for Lipogram {
    fn get_lints(&self) -> LintArray {
        lint_array!(letter_e)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let name = token::get_ident(it.ident);
        if name.get().contains_char('e') || name.get().contains_char('E') {
            cx.span_lint(letter_e, it.span, "item name contains the letter 'e'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_lint_pass(box Lipogram as LintPassObject);
}
~~~

A pass which defines multiple lints will have e.g. `lint_array!(deprecated, experimental, unstable)`.

To use a lint when compiling another crate:

~~~ .rs
#![feature(phase)]

#[phase(plugin)]
extern crate lipogram;

fn hello() { }

fn main() { hello() }
~~~

And you will get

~~~
test.rs:6:1: 6:15 warning: item name contains the letter 'e', #[warn(letter_e)] on by default
test.rs:6 fn hello() { }
          ^~~~~~~~~~~~~~
~~~

Internally, lints are identified by the address of a static `Lint`.  This has a number of benefits:

* The linker takes care of assigning unique IDs, even with dynamically loaded plugins.
* A typo writing a lint ID is usually a compiler error, unlike with string IDs.
* The ability to output a given lint is controlled by the usual visibility mechanism.  Lints defined within `rustc` use the same infrastructure and will simply export their `Lint`s if other parts of the compiler need to output those lints.
* IDs are small and easy to hash.
* It's easy to go from an ID to name, description, etc.

User-defined lints are controlled through the usual mechanism of attributes and the `-A -W -D -F` flags to `rustc`.  User-defined lints will show up in `-W help` if a crate filename is also provided; otherwise we append a message suggesting to re-run with a crate filename.

See also the [full demo](https://gist.github.com/kmcallister/3409ece44ead6d280b8e).

# Drawbacks

This increases the amount of code in `rustc` to implement lints, although it makes each individual lint much easier to understand in isolation.

Loadable lints produce more coupling of user code to `rustc` internals (with no official stability guarantee, of course).

There's no scoping / namespacing of the lint name strings used by attributes and compiler flags.  Attempting to register a lint with a duplicate name is an error at registration time.

The use of `&'static` means that lint plugins can't dynamically generate the set of lints based on some external resource.

# Alternatives

We could provide a more generic mechanism for user-defined AST visitors.  This could support other use cases like code transformation.  But it would be harder to use, and harder to integrate with the lint infrastructure.

It would be nice to magically find all static `Lint`s in a crate, so we don't need `get_lints`.  Is this worth adding another attribute and another crate metadata type?  The `plugin::Registry` mechanism was meant to avoid such a proliferation of metadata types, but it's not as declarative as I would like.

# Unresolved questions

Do we provide guarantees about visit order for a lint, or the order of multiple lints defined in the same crate?  Some lints may require multiple passes.

Should we enforce (while running lints) that each lint printed with `span_lint` was registered by the corresponding `LintPass`?  Users who particularly care can already wrap lints in modules and use visibility to enforce this statically.

Should we separate registering a lint pass from initializing / constructing the value implementing `LintPass`?  This would support a future where a single `rustc` invocation can compile multiple crates and needs to reset lint state.
