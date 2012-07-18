// Detecting language items.
//
// Language items are items that represent concepts intrinsic to the language
// itself. Examples are:
//
// * Traits that specify "kinds"; e.g. "const", "copy", "send".
//
// * Traits that represent operators; e.g. "add", "sub", "index".
//
// * Functions called by the compiler itself.

import driver::session::session;
import metadata::csearch::{each_path, get_item_attrs};
import metadata::cstore::{iter_crate_data};
import metadata::decoder::{dl_def, dl_field, dl_impl};
import syntax::ast::{crate, def_id, def_ty, lit_str, meta_item, meta_list};
import syntax::ast::{meta_name_value, meta_word};
import syntax::ast_util::{local_def};
import syntax::visit::{default_simple_visitor, mk_simple_visitor};
import syntax::visit::{visit_crate, visit_item};

import std::map::{hashmap, str_hash};
import str_eq = str::eq;

struct LanguageItems {
    let mut const_trait: option<def_id>;
    let mut copy_trait: option<def_id>;
    let mut send_trait: option<def_id>;
    let mut owned_trait: option<def_id>;

    let mut add_trait: option<def_id>;
    let mut sub_trait: option<def_id>;
    let mut mul_trait: option<def_id>;
    let mut div_trait: option<def_id>;
    let mut modulo_trait: option<def_id>;
    let mut neg_trait: option<def_id>;
    let mut bitxor_trait: option<def_id>;
    let mut bitand_trait: option<def_id>;
    let mut bitor_trait: option<def_id>;
    let mut shl_trait: option<def_id>;
    let mut shr_trait: option<def_id>;
    let mut index_trait: option<def_id>;

    new() {
        self.const_trait = none;
        self.copy_trait = none;
        self.send_trait = none;
        self.owned_trait = none;

        self.add_trait = none;
        self.sub_trait = none;
        self.mul_trait = none;
        self.div_trait = none;
        self.modulo_trait = none;
        self.neg_trait = none;
        self.bitxor_trait = none;
        self.bitand_trait = none;
        self.bitor_trait = none;
        self.shl_trait = none;
        self.shr_trait = none;
        self.index_trait = none;
    }
}

struct LanguageItemCollector {
    let items: &LanguageItems;

    let crate: @crate;
    let session: session;

    let item_refs: hashmap<~str,&mut option<def_id>>;

    new(crate: @crate, session: session, items: &self/LanguageItems) {
        self.crate = crate;
        self.session = session;
        self.items = items;
        self.item_refs = str_hash();

        self.item_refs.insert(~"const", &mut self.items.const_trait);
        self.item_refs.insert(~"copy", &mut self.items.copy_trait);
        self.item_refs.insert(~"send", &mut self.items.send_trait);
        self.item_refs.insert(~"owned", &mut self.items.owned_trait);

        self.item_refs.insert(~"add", &mut self.items.add_trait);
        self.item_refs.insert(~"sub", &mut self.items.sub_trait);
        self.item_refs.insert(~"mul", &mut self.items.mul_trait);
        self.item_refs.insert(~"div", &mut self.items.div_trait);
        self.item_refs.insert(~"modulo", &mut self.items.modulo_trait);
        self.item_refs.insert(~"neg", &mut self.items.neg_trait);
        self.item_refs.insert(~"bitxor", &mut self.items.bitxor_trait);
        self.item_refs.insert(~"bitand", &mut self.items.bitand_trait);
        self.item_refs.insert(~"bitor", &mut self.items.bitor_trait);
        self.item_refs.insert(~"shl", &mut self.items.shl_trait);
        self.item_refs.insert(~"shr", &mut self.items.shr_trait);
        self.item_refs.insert(~"index", &mut self.items.index_trait);
    }

    fn match_and_collect_meta_item(item_def_id: def_id,
                                   meta_item: meta_item) {

        match meta_item.node {
            meta_name_value(key, literal) => {
                match literal.node {
                    lit_str(value) => {
                        self.match_and_collect_item(item_def_id, key, *value);
                    }
                    _ => {} // Skip.
                }
            }
            meta_word(*) | meta_list(*) => {} // Skip.
        }
    }

    fn match_and_collect_item(item_def_id: def_id, key: ~str, value: ~str) {
        if key != ~"lang" {
            return;    // Didn't match.
        }

        match self.item_refs.find(value) {
            none => {
                // Didn't match.
            }
            some(item_ref) => {
                // Check for duplicates.
                match copy *item_ref {
                    some(original_def_id)
                            if original_def_id != item_def_id => {

                        self.session.err(fmt!{"duplicate entry for `%s`",
                                              value});
                    }
                    some(_) | none => {
                        // OK.
                    }
                }

                // Matched.
                *item_ref = some(item_def_id);
            }
        }
    }

    fn collect_local_language_items() {
        let this = unsafe { ptr::addr_of(self) };
        visit_crate(*self.crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                for item.attrs.each |attribute| {
                    unsafe {
                        (*this).match_and_collect_meta_item(local_def(item
                                                                      .id),
                                                            attribute.node
                                                                     .value);
                    }
                }
            }
            with *default_simple_visitor()
        }));
    }

    fn collect_external_language_items() {
        let crate_store = self.session.cstore;
        do iter_crate_data(crate_store) |crate_number, _crate_metadata| {
            for each_path(crate_store, crate_number) |path_entry| {
                let def_id;
                match path_entry.def_like {
                    dl_def(def_ty(did)) => {
                        def_id = did;
                    }
                    dl_def(_) | dl_impl(_) | dl_field => {
                        // Skip this.
                        again;
                    }
                }

                do get_item_attrs(crate_store, def_id) |meta_items| {
                    for meta_items.each |meta_item| {
                        self.match_and_collect_meta_item(def_id, *meta_item);
                    }
                }
            }
        }
    }

    fn check_completeness() {
        for self.item_refs.each |key, item_ref| {
            match copy *item_ref {
                none => {
                    self.session.err(fmt!{"no item found for `%s`", key});
                }
                some(did) => {
                    // OK.
                }
            }
        }
    }

    fn collect() {
        self.collect_local_language_items();
        self.collect_external_language_items();
        self.check_completeness();
    }
}

fn collect_language_items(crate: @crate, session: session) -> LanguageItems {
    let items = LanguageItems();
    let collector = LanguageItemCollector(crate, session, &items);
    collector.collect();
    copy items
}

