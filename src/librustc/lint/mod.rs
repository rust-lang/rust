// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints, aka compiler warnings.
//!
//! A 'lint' check is a kind of miscellaneous constraint that a user _might_
//! want to enforce, but might reasonably want to permit as well, on a
//! module-by-module basis. They contrast with static constraints enforced by
//! other phases of the compiler, which are generally required to hold in order
//! to compile the program at all.
//!
//! Most lints can be written as `LintPass` instances. These run just before
//! translation to LLVM bytecode. The `LintPass`es built into rustc are defined
//! within `builtin.rs`, which has further comments on how to add such a lint.
//! rustc can also load user-defined lint plugins via the plugin mechanism.
//!
//! Some of rustc's lints are defined elsewhere in the compiler and work by
//! calling `add_lint()` on the overall `Session` object. This works when
//! it happens before the main lint pass, which emits the lints stored by
//! `add_lint()`. To emit lints after the main lint pass (from trans, for
//! example) requires more effort. See `emit_lint` and `GatherNodeLevels`
//! in `context.rs`.

#![macro_escape]

use std::hash;
use std::ascii::AsciiExt;
use syntax::codemap::Span;
use syntax::visit::FnKind;
use syntax::ast;

pub use lint::context::{Context, LintStore, raw_emit_lint, check_crate, gather_attrs};

/// Specification of a single lint.
pub struct Lint {
    /// A string identifier for the lint.
    ///
    /// This identifies the lint in attributes and in command-line arguments.
    /// In those contexts it is always lowercase, but this field is compared
    /// in a way which is case-insensitive for ASCII characters. This allows
    /// `declare_lint!()` invocations to follow the convention of upper-case
    /// statics without repeating the name.
    ///
    /// The name is written with underscores, e.g. "unused_imports".
    /// On the command line, underscores become dashes.
    pub name: &'static str,

    /// Default level for the lint.
    pub default_level: Level,

    /// Description of the lint or the issue it detects.
    ///
    /// e.g. "imports that are never used"
    pub desc: &'static str,
}

impl Lint {
    /// Get the lint's name, with ASCII letters converted to lowercase.
    pub fn name_lower(&self) -> String {
        self.name.to_ascii_lower()
    }
}

/// Build a `Lint` initializer.
#[macro_export]
macro_rules! lint_initializer (
    ($name:ident, $level:ident, $desc:expr) => (
        ::rustc::lint::Lint {
            name: stringify!($name),
            default_level: ::rustc::lint::$level,
            desc: $desc,
        }
    )
)

/// Declare a static item of type `&'static Lint`.
#[macro_export]
macro_rules! declare_lint (
    // FIXME(#14660): deduplicate
    (pub $name:ident, $level:ident, $desc:expr) => (
        pub static $name: &'static ::rustc::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    );
    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static ::rustc::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    );
)

/// Declare a static `LintArray` and return it as an expression.
#[macro_export]
macro_rules! lint_array ( ($( $lint:expr ),*) => (
    {
        static array: LintArray = &[ $( $lint ),* ];
        array
    }
))

pub type LintArray = &'static [&'static Lint];

/// Trait for types providing lint checks.
///
/// Each `check` method checks a single syntax node, and should not
/// invoke methods recursively (unlike `Visitor`). By default they
/// do nothing.
//
// FIXME: eliminate the duplication with `Visitor`. But this also
// contains a few lint-specific methods with no equivalent in `Visitor`.
pub trait LintPass {
    /// Get descriptions of the lints this `LintPass` object can emit.
    ///
    /// NB: there is no enforcement that the object only emits lints it registered.
    /// And some `rustc` internal `LintPass`es register lints to be emitted by other
    /// parts of the compiler. If you want enforced access restrictions for your
    /// `Lint`, make it a private `static` item in its own module.
    fn get_lints(&self) -> LintArray;

    fn check_crate(&mut self, _: &Context, _: &ast::Crate) { }
    fn check_ident(&mut self, _: &Context, _: Span, _: ast::Ident) { }
    fn check_mod(&mut self, _: &Context, _: &ast::Mod, _: Span, _: ast::NodeId) { }
    fn check_view_item(&mut self, _: &Context, _: &ast::ViewItem) { }
    fn check_foreign_item(&mut self, _: &Context, _: &ast::ForeignItem) { }
    fn check_item(&mut self, _: &Context, _: &ast::Item) { }
    fn check_local(&mut self, _: &Context, _: &ast::Local) { }
    fn check_block(&mut self, _: &Context, _: &ast::Block) { }
    fn check_stmt(&mut self, _: &Context, _: &ast::Stmt) { }
    fn check_arm(&mut self, _: &Context, _: &ast::Arm) { }
    fn check_pat(&mut self, _: &Context, _: &ast::Pat) { }
    fn check_decl(&mut self, _: &Context, _: &ast::Decl) { }
    fn check_expr(&mut self, _: &Context, _: &ast::Expr) { }
    fn check_expr_post(&mut self, _: &Context, _: &ast::Expr) { }
    fn check_ty(&mut self, _: &Context, _: &ast::Ty) { }
    fn check_generics(&mut self, _: &Context, _: &ast::Generics) { }
    fn check_fn(&mut self, _: &Context,
        _: &FnKind, _: &ast::FnDecl, _: &ast::Block, _: Span, _: ast::NodeId) { }
    fn check_ty_method(&mut self, _: &Context, _: &ast::TypeMethod) { }
    fn check_trait_method(&mut self, _: &Context, _: &ast::TraitMethod) { }
    fn check_struct_def(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, _: ast::NodeId) { }
    fn check_struct_def_post(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, _: ast::NodeId) { }
    fn check_struct_field(&mut self, _: &Context, _: &ast::StructField) { }
    fn check_variant(&mut self, _: &Context, _: &ast::Variant, _: &ast::Generics) { }
    fn check_opt_lifetime_ref(&mut self, _: &Context, _: Span, _: &Option<ast::Lifetime>) { }
    fn check_lifetime_ref(&mut self, _: &Context, _: &ast::Lifetime) { }
    fn check_lifetime_decl(&mut self, _: &Context, _: &ast::LifetimeDef) { }
    fn check_explicit_self(&mut self, _: &Context, _: &ast::ExplicitSelf) { }
    fn check_mac(&mut self, _: &Context, _: &ast::Mac) { }
    fn check_path(&mut self, _: &Context, _: &ast::Path, _: ast::NodeId) { }
    fn check_attribute(&mut self, _: &Context, _: &ast::Attribute) { }

    /// Called when entering a syntax node that can have lint attributes such
    /// as `#[allow(...)]`. Called with *all* the attributes of that node.
    fn enter_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) { }

    /// Counterpart to `enter_lint_attrs`.
    fn exit_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) { }
}

/// A lint pass boxed up as a trait object.
pub type LintPassObject = Box<LintPass + 'static>;

/// Identifies a lint known to the compiler.
#[deriving(Clone)]
pub struct LintId {
    // Identity is based on pointer equality of this field.
    lint: &'static Lint,
}

impl PartialEq for LintId {
    fn eq(&self, other: &LintId) -> bool {
        (self.lint as *const Lint) == (other.lint as *const Lint)
    }
}

impl Eq for LintId { }

impl<S: hash::Writer> hash::Hash<S> for LintId {
    fn hash(&self, state: &mut S) {
        let ptr = self.lint as *const Lint;
        ptr.hash(state);
    }
}

impl LintId {
    /// Get the `LintId` for a `Lint`.
    pub fn of(lint: &'static Lint) -> LintId {
        LintId {
            lint: lint,
        }
    }

    /// Get the name of the lint.
    pub fn as_str(&self) -> String {
        self.lint.name_lower()
    }
}

/// Setting for how to handle a lint.
#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Level {
    Allow, Warn, Deny, Forbid
}

impl Level {
    /// Convert a level to a lower-case string.
    pub fn as_str(self) -> &'static str {
        match self {
            Allow => "allow",
            Warn => "warn",
            Deny => "deny",
            Forbid => "forbid",
        }
    }

    /// Convert a lower-case string to a level.
    pub fn from_str(x: &str) -> Option<Level> {
        match x {
            "allow" => Some(Allow),
            "warn" => Some(Warn),
            "deny" => Some(Deny),
            "forbid" => Some(Forbid),
            _ => None,
        }
    }
}

/// How a lint level was set.
#[deriving(Clone, PartialEq, Eq)]
pub enum LintSource {
    /// Lint is at the default level as declared
    /// in rustc or a plugin.
    Default,

    /// Lint level was set by an attribute.
    Node(Span),

    /// Lint level was set by a command-line flag.
    CommandLine,
}

pub type LevelSource = (Level, LintSource);

pub mod builtin;

mod context;
