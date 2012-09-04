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

use driver::session::session;
use metadata::csearch::{each_path, get_item_attrs};
use metadata::cstore::{iter_crate_data};
use metadata::decoder::{dl_def, dl_field, dl_impl};
use syntax::ast::{crate, def_id, def_ty, lit_str, meta_item, meta_list};
use syntax::ast::{meta_name_value, meta_word};
use syntax::ast_util::{local_def};
use syntax::visit::{default_simple_visitor, mk_simple_visitor};
use syntax::visit::{visit_crate, visit_item};

use std::map::{hashmap, str_hash};
use str_eq = str::eq;

struct LanguageItems {
    mut const_trait: Option<def_id>;
    mut copy_trait: Option<def_id>;
    mut send_trait: Option<def_id>;
    mut owned_trait: Option<def_id>;

    mut add_trait: Option<def_id>;
    mut sub_trait: Option<def_id>;
    mut mul_trait: Option<def_id>;
    mut div_trait: Option<def_id>;
    mut modulo_trait: Option<def_id>;
    mut neg_trait: Option<def_id>;
    mut bitxor_trait: Option<def_id>;
    mut bitand_trait: Option<def_id>;
    mut bitor_trait: Option<def_id>;
    mut shl_trait: Option<def_id>;
    mut shr_trait: Option<def_id>;
    mut index_trait: Option<def_id>;

    mut eq_trait: Option<def_id>;
    mut ord_trait: Option<def_id>;
}

mod LanguageItems {
    fn make() -> LanguageItems {
        LanguageItems {
            const_trait: None,
            copy_trait: None,
            send_trait: None,
            owned_trait: None,

            add_trait: None,
            sub_trait: None,
            mul_trait: None,
            div_trait: None,
            modulo_trait: None,
            neg_trait: None,
            bitxor_trait: None,
            bitand_trait: None,
            bitor_trait: None,
            shl_trait: None,
            shr_trait: None,
            index_trait: None,

            eq_trait: None,
            ord_trait: None
        }
    }
}

struct LanguageItemCollector {
    let items: &LanguageItems;

    let crate: @crate;
    let session: session;

    let item_refs: hashmap<~str,&mut Option<def_id>>;

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

        self.item_refs.insert(~"eq", &mut self.items.eq_trait);
        self.item_refs.insert(~"ord", &mut self.items.ord_trait);
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
            None => {
                // Didn't match.
            }
            Some(item_ref) => {
                // Check for duplicates.
                match copy *item_ref {
                    Some(original_def_id)
                            if original_def_id != item_def_id => {

                        self.session.err(fmt!("duplicate entry for `%s`",
                                              value));
                    }
                    Some(_) | None => {
                        // OK.
                    }
                }

                // Matched.
                *item_ref = Some(item_def_id);
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
            },
            .. *default_simple_visitor()
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
                None => {
                    self.session.err(fmt!("no item found for `%s`", key));
                }
                Some(_) => {
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
    let items = LanguageItems::make();
    let collector = LanguageItemCollector(crate, session, &items);
    collector.collect();
    copy items
}

