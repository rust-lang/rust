// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Used by `rustc` when loading a crate with exported macros.

use session::Session;
use metadata::creader::CrateReader;

use std::collections::{HashSet, HashMap};
use syntax::ast;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::attr::AttrMetaMethods;

struct MacroLoader<'a> {
    sess: &'a Session,
    span_whitelist: HashSet<Span>,
    reader: CrateReader<'a>,
    macros: Vec<ast::MacroDef>,
}

impl<'a> MacroLoader<'a> {
    fn new(sess: &'a Session) -> MacroLoader<'a> {
        MacroLoader {
            sess: sess,
            span_whitelist: HashSet::new(),
            reader: CrateReader::new(sess),
            macros: vec![],
        }
    }
}

/// Read exported macros.
pub fn read_macro_defs(sess: &Session, krate: &ast::Crate) -> Vec<ast::MacroDef> {
    let mut loader = MacroLoader::new(sess);

    // We need to error on `#[macro_use] extern crate` when it isn't at the
    // crate root, because `$crate` won't work properly. Identify these by
    // spans, because the crate map isn't set up yet.
    for item in &krate.module.items {
        if let ast::ItemExternCrate(_) = item.node {
            loader.span_whitelist.insert(item.span);
        }
    }

    visit::walk_crate(&mut loader, krate);

    loader.macros
}

pub type MacroSelection = HashMap<token::InternedString, Span>;

// note that macros aren't expanded yet, and therefore macros can't add macro imports.
impl<'a, 'v> Visitor<'v> for MacroLoader<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        // We're only interested in `extern crate`.
        match item.node {
            ast::ItemExternCrate(_) => {}
            _ => {
                visit::walk_item(self, item);
                return;
            }
        }

        // Parse the attributes relating to macros.
        let mut import = Some(HashMap::new());  // None => load all
        let mut reexport = HashMap::new();

        for attr in &item.attrs {
            let mut used = true;
            match &attr.name()[..] {
                "macro_use" => {
                    let names = attr.meta_item_list();
                    if names.is_none() {
                        // no names => load all
                        import = None;
                    }
                    if let (Some(sel), Some(names)) = (import.as_mut(), names) {
                        for attr in names {
                            if let ast::MetaWord(ref name) = attr.node {
                                sel.insert(name.clone(), attr.span);
                            } else {
                                self.sess.span_err(attr.span, "bad macro import");
                            }
                        }
                    }
                }
                "macro_reexport" => {
                    let names = match attr.meta_item_list() {
                        Some(names) => names,
                        None => {
                            self.sess.span_err(attr.span, "bad macro reexport");
                            continue;
                        }
                    };

                    for attr in names {
                        if let ast::MetaWord(ref name) = attr.node {
                            reexport.insert(name.clone(), attr.span);
                        } else {
                            self.sess.span_err(attr.span, "bad macro reexport");
                        }
                    }
                }
                _ => used = false,
            }
            if used {
                attr::mark_used(attr);
            }
        }

        self.load_macros(item, import, reexport)
    }

    fn visit_mac(&mut self, _: &ast::Mac) {
        // bummer... can't see macro imports inside macros.
        // do nothing.
    }
}

impl<'a> MacroLoader<'a> {
    fn load_macros<'b>(&mut self,
                       vi: &ast::Item,
                       import: Option<MacroSelection>,
                       reexport: MacroSelection) {
        if let Some(sel) = import.as_ref() {
            if sel.is_empty() && reexport.is_empty() {
                return;
            }
        }

        if !self.span_whitelist.contains(&vi.span) {
            self.sess.span_err(vi.span, "an `extern crate` loading macros must be at \
                                         the crate root");
            return;
        }

        let macros = self.reader.read_exported_macros(vi);
        let mut seen = HashSet::new();

        for mut def in macros {
            let name = token::get_ident(def.ident);
            seen.insert(name.clone());

            def.use_locally = match import.as_ref() {
                None => true,
                Some(sel) => sel.contains_key(&name),
            };
            def.export = reexport.contains_key(&name);
            def.allow_internal_unstable = attr::contains_name(&def.attrs,
                                                              "allow_internal_unstable");
            debug!("load_macros: loaded: {:?}", def);
            self.macros.push(def);
        }

        if let Some(sel) = import.as_ref() {
            for (name, span) in sel.iter() {
                if !seen.contains(name) {
                    self.sess.span_err(*span, "imported macro not found");
                }
            }
        }

        for (name, span) in reexport.iter() {
            if !seen.contains(name) {
                self.sess.span_err(*span, "reexported macro not found");
            }
        }
    }
}
