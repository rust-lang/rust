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

use std::collections::HashSet;
use std::rc::Rc;
use std::env;
use std::mem;

use creader::{CrateLoader, Macros};

use proc_macro::TokenStream;
use proc_macro::__internal::Registry;
use rustc::hir::def_id::DefIndex;
use rustc::middle::cstore::{LoadedMacro, LoadedMacroKind};
use rustc::session::Session;
use rustc::util::nodemap::FnvHashMap;
use rustc_back::dynamic_lib::DynamicLibrary;
use syntax::ast;
use syntax::attr;
use syntax::parse::token;
use syntax_ext::deriving::custom::CustomDerive;
use syntax_pos::{Span, DUMMY_SP};

pub fn call_bad_macro_reexport(a: &Session, b: Span) {
    span_err!(a, b, E0467, "bad macro reexport");
}

pub type MacroSelection = FnvHashMap<token::InternedString, Span>;

enum ImportSelection {
    All(Span),
    Some(MacroSelection),
}

pub fn load_macros(loader: &mut CrateLoader, extern_crate: &ast::Item, allows_macros: bool)
                   -> Vec<LoadedMacro> {
    loader.load_crate(extern_crate, allows_macros)
}

impl<'a> CrateLoader<'a> {
    fn load_crate(&mut self,
                  extern_crate: &ast::Item,
                  allows_macros: bool) -> Vec<LoadedMacro> {
        // Parse the attributes relating to macros.
        let mut import = ImportSelection::Some(FnvHashMap());
        let mut reexport = FnvHashMap();

        for attr in &extern_crate.attrs {
            let mut used = true;
            match &attr.name()[..] {
                "macro_use" => {
                    let names = attr.meta_item_list();
                    if names.is_none() {
                        import = ImportSelection::All(attr.span);
                    } else if let ImportSelection::Some(ref mut sel) = import {
                        for attr in names.unwrap() {
                            if let Some(word) = attr.word() {
                                sel.insert(word.name().clone(), attr.span());
                            } else {
                                span_err!(self.sess, attr.span(), E0466, "bad macro import");
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
                        if let Some(word) = attr.word() {
                            reexport.insert(word.name().clone(), attr.span());
                        } else {
                            call_bad_macro_reexport(self.sess, attr.span());
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

    fn load_macros<'b>(&mut self,
                       vi: &ast::Item,
                       allows_macros: bool,
                       import: ImportSelection,
                       reexport: MacroSelection)
                       -> Vec<LoadedMacro> {
        if let ImportSelection::Some(ref sel) = import {
            if sel.is_empty() && reexport.is_empty() {
                return Vec::new();
            }
        }

        if !allows_macros {
            span_err!(self.sess, vi.span, E0468,
                      "an `extern crate` loading macros must be at the crate root");
            return Vec::new();
        }

        let mut macros = self.creader.read_macros(vi);
        let mut ret = Vec::new();
        let mut seen = HashSet::new();

        for mut def in macros.macro_rules.drain(..) {
            let name = def.ident.name.as_str();

            let import_site = match import {
                ImportSelection::All(span) => Some(span),
                ImportSelection::Some(ref sel) => sel.get(&name).cloned()
            };
            def.use_locally = import_site.is_some();
            def.export = reexport.contains_key(&name);
            def.allow_internal_unstable = attr::contains_name(&def.attrs,
                                                              "allow_internal_unstable");
            debug!("load_macros: loaded: {:?}", def);
            ret.push(LoadedMacro {
                kind: LoadedMacroKind::Def(def),
                import_site: import_site.unwrap_or(DUMMY_SP),
            });
            seen.insert(name);
        }

        if let Some(index) = macros.custom_derive_registrar {
            // custom derive crates currently should not have any macro_rules!
            // exported macros, enforced elsewhere
            assert_eq!(ret.len(), 0);

            if let ImportSelection::Some(..) = import {
                self.sess.span_err(vi.span, "`proc-macro` crates cannot be \
                                             selectively imported from, must \
                                             use `#[macro_use]`");
            }

            if reexport.len() > 0 {
                self.sess.span_err(vi.span, "`proc-macro` crates cannot be \
                                             reexported from");
            }

            self.load_derive_macros(vi.span, &macros, index, &mut ret);
        }

        if let ImportSelection::Some(sel) = import {
            for (name, span) in sel {
                if !seen.contains(&name) {
                    span_err!(self.sess, span, E0469,
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

        return ret
    }

    /// Load the custom derive macros into the list of macros we're loading.
    ///
    /// Note that this is intentionally similar to how we load plugins today,
    /// but also intentionally separate. Plugins are likely always going to be
    /// implemented as dynamic libraries, but we have a possible future where
    /// custom derive (and other macro-1.1 style features) are implemented via
    /// executables and custom IPC.
    fn load_derive_macros(&mut self,
                          span: Span,
                          macros: &Macros,
                          index: DefIndex,
                          ret: &mut Vec<LoadedMacro>) {
        // Make sure the path contains a / or the linker will search for it.
        let path = macros.dylib.as_ref().unwrap();
        let path = env::current_dir().unwrap().join(path);
        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            Err(err) => self.sess.span_fatal(span, &err),
        };

        let sym = self.sess.generate_derive_registrar_symbol(&macros.svh, index);
        let registrar = unsafe {
            let sym = match lib.symbol(&sym) {
                Ok(f) => f,
                Err(err) => self.sess.span_fatal(span, &err),
            };
            mem::transmute::<*mut u8, fn(&mut Registry)>(sym)
        };

        struct MyRegistrar<'a>(&'a mut Vec<LoadedMacro>, Span);

        impl<'a> Registry for MyRegistrar<'a> {
            fn register_custom_derive(&mut self,
                                      trait_name: &str,
                                      expand: fn(TokenStream) -> TokenStream) {
                let derive = Rc::new(CustomDerive::new(expand));
                self.0.push(LoadedMacro {
                    kind: LoadedMacroKind::CustomDerive(trait_name.to_string(), derive),
                    import_site: self.1,
                });
            }
        }

        registrar(&mut MyRegistrar(ret, span));

        // Intentionally leak the dynamic library. We can't ever unload it
        // since the library can make things that will live arbitrarily long.
        mem::forget(lib);
    }
}
