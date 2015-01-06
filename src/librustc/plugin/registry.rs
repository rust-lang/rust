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

use lint::{LintPassObject, LintId, Lint};
use session::Session;

use syntax::ext::base::{SyntaxExtension, NamedSyntaxExtension, NormalTT};
use syntax::ext::base::{IdentTT, Decorator, Modifier, MacroRulesTT};
use syntax::ext::base::{MacroExpanderFn};
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::ast;

use std::collections::HashMap;

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
    pub args_hidden: Option<P<ast::MetaItem>>,

    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub syntax_exts: Vec<NamedSyntaxExtension>,

    #[doc(hidden)]
    pub lint_passes: Vec<LintPassObject>,

    #[doc(hidden)]
    pub lint_groups: HashMap<&'static str, Vec<LintId>>,
}

impl<'a> Registry<'a> {
    #[doc(hidden)]
    pub fn new(sess: &'a Session, krate: &ast::Crate) -> Registry<'a> {
        Registry {
            sess: sess,
            args_hidden: None,
            krate_span: krate.span,
            syntax_exts: vec!(),
            lint_passes: vec!(),
            lint_groups: HashMap::new(),
        }
    }

    /// Get the `#[plugin]` attribute used to load this plugin.
    ///
    /// This gives access to arguments passed via `#[plugin=...]` or
    /// `#[plugin(...)]`.
    pub fn args<'b>(&'b self) -> &'b P<ast::MetaItem> {
        self.args_hidden.as_ref().expect("args not set")
    }

    /// Register a syntax extension of any kind.
    ///
    /// This is the most general hook into `libsyntax`'s expansion behavior.
    pub fn register_syntax_extension(&mut self, name: ast::Name, extension: SyntaxExtension) {
        self.syntax_exts.push((name, match extension {
            NormalTT(ext, _) => NormalTT(ext, Some(self.krate_span)),
            IdentTT(ext, _) => IdentTT(ext, Some(self.krate_span)),
            Decorator(ext) => Decorator(ext),
            Modifier(ext) => Modifier(ext),

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
        self.register_syntax_extension(token::intern(name), NormalTT(box expander, None));
    }

    /// Register a compiler lint pass.
    pub fn register_lint_pass(&mut self, lint_pass: LintPassObject) {
        self.lint_passes.push(lint_pass);
    }

    /// Register a lint group.
    pub fn register_lint_group(&mut self, name: &'static str, to: Vec<&'static Lint>) {
        self.lint_groups.insert(name, to.into_iter().map(|x| LintId::of(x)).collect());
    }
}
