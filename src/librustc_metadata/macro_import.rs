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

use creader::CrateReader;
use cstore::CStore;

use rustc::session::Session;

use std::collections::{HashSet, HashMap};
use syntax::parse::token;
use syntax::ast;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::ext;
use syntax_pos::Span;

pub struct MacroLoader<'a> {
    sess: &'a Session,
    reader: CrateReader<'a>,
}

impl<'a> MacroLoader<'a> {
    pub fn new(sess: &'a Session, cstore: &'a CStore, crate_name: &str) -> MacroLoader<'a> {
        MacroLoader {
            sess: sess,
            reader: CrateReader::new(sess, cstore, crate_name),
        }
    }
}

pub fn call_bad_macro_reexport(a: &Session, b: Span) {
    span_err!(a, b, E0467, "bad macro reexport");
}

pub type MacroSelection = HashMap<token::InternedString, Span>;

impl<'a> ext::base::MacroLoader for MacroLoader<'a> {
    fn load_crate(&mut self, extern_crate: &ast::Item, allows_macros: bool) -> Vec<ast::MacroDef> {
        // Parse the attributes relating to macros.
        let mut import = Some(HashMap::new());  // None => load all
        let mut reexport = HashMap::new();

        for attr in &extern_crate.attrs {
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
                            if let ast::MetaItemKind::Word(ref name) = attr.node {
                                sel.insert(name.clone(), attr.span);
                            } else {
                                span_err!(self.sess, attr.span, E0466, "bad macro import");
                            }
                        }
                    }
                }
                "macro_reexport" => {
                    let names = match attr.meta_item_list() {
                        Some(names) => names,
                        None => {
                            call_bad_macro_reexport(self.sess, attr.span);
                            continue;
                        }
                    };

                    for attr in names {
                        if let ast::MetaItemKind::Word(ref name) = attr.node {
                            reexport.insert(name.clone(), attr.span);
                        } else {
                            call_bad_macro_reexport(self.sess, attr.span);
                        }
                    }
                }
                _ => used = false,
            }
            if used {
                attr::mark_used(attr);
            }
        }

        self.load_macros(extern_crate, allows_macros, import, reexport)
    }
}

impl<'a> MacroLoader<'a> {
    fn load_macros<'b>(&mut self,
                       vi: &ast::Item,
                       allows_macros: bool,
                       import: Option<MacroSelection>,
                       reexport: MacroSelection)
                       -> Vec<ast::MacroDef> {
        if let Some(sel) = import.as_ref() {
            if sel.is_empty() && reexport.is_empty() {
                return Vec::new();
            }
        }

        if !allows_macros {
            span_err!(self.sess, vi.span, E0468,
                      "an `extern crate` loading macros must be at the crate root");
            return Vec::new();
        }

        let mut macros = Vec::new();
        let mut seen = HashSet::new();

        for mut def in self.reader.read_exported_macros(vi) {
            let name = def.ident.name.as_str();

            def.use_locally = match import.as_ref() {
                None => true,
                Some(sel) => sel.contains_key(&name),
            };
            def.export = reexport.contains_key(&name);
            def.allow_internal_unstable = attr::contains_name(&def.attrs,
                                                              "allow_internal_unstable");
            debug!("load_macros: loaded: {:?}", def);
            macros.push(def);
            seen.insert(name);
        }

        if let Some(sel) = import.as_ref() {
            for (name, span) in sel {
                if !seen.contains(&name) {
                    span_err!(self.sess, *span, E0469,
                              "imported macro not found");
                }
            }
        }

        for (name, span) in &reexport {
            if !seen.contains(&name) {
                span_err!(self.sess, *span, E0470,
                          "reexported macro not found");
            }
        }

        macros
    }
}
