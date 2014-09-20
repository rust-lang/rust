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

use syntax::ext::base::{SyntaxExtension, NamedSyntaxExtension, NormalTT};
use syntax::ext::base::{IdentTT, LetSyntaxTT, Decorator, Modifier};
use syntax::ext::base::{MacroExpanderFn};
use syntax::codemap::Span;
use syntax::parse::token;
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
pub struct Registry {
    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub syntax_exts: Vec<NamedSyntaxExtension>,

    #[doc(hidden)]
    pub lint_passes: Vec<LintPassObject>,

    #[doc(hidden)]
    pub lint_groups: HashMap<&'static str, Vec<LintId>>,
}

impl Registry {
    #[doc(hidden)]
    pub fn new(krate: &ast::Crate) -> Registry {
        Registry {
            krate_span: krate.span,
            syntax_exts: vec!(),
            lint_passes: vec!(),
            lint_groups: HashMap::new(),
        }
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
            // there's probably a nicer way to signal this:
            LetSyntaxTT(_, _) => fail!("can't register a new LetSyntax!"),
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
