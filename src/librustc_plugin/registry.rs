// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Used by plugin crates to tell `rustc` about the plugins they provide.

use rustc::lint::{EarlyLintPassObject, LateLintPassObject, LintId, Lint};
use rustc::session::Session;

use syntax::ext::base::{SyntaxExtension, NamedSyntaxExtension, NormalTT};
use syntax::ext::base::{IdentTT, MultiModifier, MultiDecorator};
use syntax::ext::base::{MacroExpanderFn, MacroRulesTT};
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::ast;
use syntax::feature_gate::AttributeType;

use std::collections::HashMap;
use std::borrow::ToOwned;

/// Structure used to register plugins.
///
/// A plugin registrar function takes an `&mut Registry` and should call
/// methods to register its plugins.
///
/// This struct has public fields and other methods for use by `rustc`
/// itself. They are not documented here, and plugin authors should
/// not use them.
pub struct Registry<'a> {
    /// Compiler session. Useful if you want to emit diagnostic messages
    /// from the plugin registrar.
    pub sess: &'a Session,

    #[doc(hidden)]
    pub args_hidden: Option<Vec<P<ast::MetaItem>>>,

    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub syntax_exts: Vec<NamedSyntaxExtension>,

    #[doc(hidden)]
    pub early_lint_passes: Vec<EarlyLintPassObject>,

    #[doc(hidden)]
    pub late_lint_passes: Vec<LateLintPassObject>,

    #[doc(hidden)]
    pub lint_groups: HashMap<&'static str, Vec<LintId>>,

    #[doc(hidden)]
    pub llvm_passes: Vec<String>,

    #[doc(hidden)]
    pub attributes: Vec<(String, AttributeType)>,
}

impl<'a> Registry<'a> {
    #[doc(hidden)]
    pub fn new(sess: &'a Session, krate: &ast::Crate) -> Registry<'a> {
        Registry {
            sess: sess,
            args_hidden: None,
            krate_span: krate.span,
            syntax_exts: vec!(),
            early_lint_passes: vec!(),
            late_lint_passes: vec!(),
            lint_groups: HashMap::new(),
            llvm_passes: vec!(),
            attributes: vec!(),
        }
    }

    /// Get the plugin's arguments, if any.
    ///
    /// These are specified inside the `plugin` crate attribute as
    ///
    /// ```no_run
    /// #![plugin(my_plugin_name(... args ...))]
    /// ```
    pub fn args<'b>(&'b self) -> &'b Vec<P<ast::MetaItem>> {
        self.args_hidden.as_ref().expect("args not set")
    }

    /// Register a syntax extension of any kind.
    ///
    /// This is the most general hook into `libsyntax`'s expansion behavior.
    pub fn register_syntax_extension(&mut self, name: ast::Name, extension: SyntaxExtension) {
        self.syntax_exts.push((name, match extension {
            NormalTT(ext, _, allow_internal_unstable) => {
                NormalTT(ext, Some(self.krate_span), allow_internal_unstable)
            }
            IdentTT(ext, _, allow_internal_unstable) => {
                IdentTT(ext, Some(self.krate_span), allow_internal_unstable)
            }
            MultiDecorator(ext) => MultiDecorator(ext),
            MultiModifier(ext) => MultiModifier(ext),
            MacroRulesTT => {
                self.sess.err("plugin tried to register a new MacroRulesTT");
                return;
            }
        }));
    }

    /// Register a macro of the usual kind.
    ///
    /// This is a convenience wrapper for `register_syntax_extension`.
    /// It builds for you a `NormalTT` that calls `expander`,
    /// and also takes care of interning the macro's name.
    pub fn register_macro(&mut self, name: &str, expander: MacroExpanderFn) {
        self.register_syntax_extension(token::intern(name),
                                       NormalTT(Box::new(expander), None, false));
    }

    /// Register a compiler lint pass.
    pub fn register_early_lint_pass(&mut self, lint_pass: EarlyLintPassObject) {
        self.early_lint_passes.push(lint_pass);
    }

    /// Register a compiler lint pass.
    pub fn register_late_lint_pass(&mut self, lint_pass: LateLintPassObject) {
        self.late_lint_passes.push(lint_pass);
    }
    /// Register a lint group.
    pub fn register_lint_group(&mut self, name: &'static str, to: Vec<&'static Lint>) {
        self.lint_groups.insert(name, to.into_iter().map(|x| LintId::of(x)).collect());
    }

    /// Register an LLVM pass.
    ///
    /// Registration with LLVM itself is handled through static C++ objects with
    /// constructors. This method simply adds a name to the list of passes to
    /// execute.
    pub fn register_llvm_pass(&mut self, name: &str) {
        self.llvm_passes.push(name.to_owned());
    }


    /// Register an attribute with an attribute type.
    ///
    /// Registered attributes will bypass the `custom_attribute` feature gate.
    /// `Whitelisted` attributes will additionally not trigger the `unused_attribute`
    /// lint. `CrateLevel` attributes will not be allowed on anything other than a crate.
    pub fn register_attribute(&mut self, name: String, ty: AttributeType) {
        self.attributes.push((name, ty));
    }
}
