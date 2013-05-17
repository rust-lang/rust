// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Attribute parsing

The attribute parser provides methods for pulling documentation out of
an AST's attributes.
*/

use core::prelude::*;

use syntax::ast;
use syntax::attr;

pub struct CrateAttrs {
    name: Option<~str>
}

fn doc_metas(
    attrs: ~[ast::attribute]
) -> ~[@ast::meta_item] {

    let doc_attrs = attr::find_attrs_by_name(attrs, "doc");
    let doc_metas = do doc_attrs.map |attr| {
        attr::attr_meta(attr::desugar_doc_attr(attr))
    };

    return doc_metas;
}

pub fn parse_crate(attrs: ~[ast::attribute]) -> CrateAttrs {
    let link_metas = attr::find_linkage_metas(attrs);
    let name = attr::last_meta_item_value_str_by_name(link_metas, "name");

    CrateAttrs {
        name: name.map(|s| copy **s)
    }
}

pub fn parse_desc(attrs: ~[ast::attribute]) -> Option<~str> {
    let doc_strs = do doc_metas(attrs).filter_mapped |meta| {
        attr::get_meta_item_value_str(*meta).map(|s| copy **s)
    };
    if doc_strs.is_empty() {
        None
    } else {
        Some(str::connect(doc_strs, "\n"))
    }
}

pub fn parse_hidden(attrs: ~[ast::attribute]) -> bool {
    do doc_metas(attrs).find |meta| {
        match attr::get_meta_item_list(*meta) {
            Some(metas) => {
                let hiddens = attr::find_meta_items_by_name(metas, "hidden");
                !hiddens.is_empty()
            }
            None => false
        }
    }.is_some()
}

#[cfg(test)]
mod test {
    use syntax::ast;
    use syntax;
    use super::{parse_hidden, parse_crate, parse_desc};

    fn parse_attributes(source: ~str) -> ~[ast::attribute] {
        use syntax::parse;
        use syntax::parse::attr::parser_attr;
        use syntax::codemap;

        let parse_sess = syntax::parse::new_parse_sess(None);
        let parser = parse::new_parser_from_source_str(
            parse_sess, ~[], ~"-", @source);

        parser.parse_outer_attributes()
    }


    #[test]
    fn should_extract_crate_name_from_link_attribute() {
        let source = ~"#[link(name = \"snuggles\")]";
        let attrs = parse_attributes(source);
        let attrs = parse_crate(attrs);
        assert!(attrs.name == Some(~"snuggles"));
    }

    #[test]
    fn should_not_extract_crate_name_if_no_link_attribute() {
        let source = ~"";
        let attrs = parse_attributes(source);
        let attrs = parse_crate(attrs);
        assert!(attrs.name == None);
    }

    #[test]
    fn should_not_extract_crate_name_if_no_name_value_in_link_attribute() {
        let source = ~"#[link(whatever)]";
        let attrs = parse_attributes(source);
        let attrs = parse_crate(attrs);
        assert!(attrs.name == None);
    }

    #[test]
    fn parse_desc_should_handle_undocumented_mods() {
        let source = ~"";
        let attrs = parse_attributes(source);
        let attrs = parse_desc(attrs);
        assert!(attrs == None);
    }

    #[test]
    fn parse_desc_should_parse_simple_doc_attributes() {
        let source = ~"#[doc = \"basic\"]";
        let attrs = parse_attributes(source);
        let attrs = parse_desc(attrs);
        assert!(attrs == Some(~"basic"));
    }

    #[test]
    fn should_parse_hidden_attribute() {
        let source = ~"#[doc(hidden)]";
        let attrs = parse_attributes(source);
        assert!(parse_hidden(attrs) == true);
    }

    #[test]
    fn should_parse_hidden_attribute_with_other_docs() {
        let source = ~"#[doc = \"foo\"] #[doc(hidden)] #[doc = \"foo\"]";
        let attrs = parse_attributes(source);
        assert!(parse_hidden(attrs) == true);
    }

    #[test]
    fn should_not_parse_non_hidden_attribute() {
        let source = ~"#[doc = \"\"]";
        let attrs = parse_attributes(source);
        assert!(parse_hidden(attrs) == false);
    }

    #[test]
    fn should_concatenate_multiple_doc_comments() {
        let source = ~"/// foo\n/// bar";
        let desc = parse_desc(parse_attributes(source));
        assert!(desc == Some(~"foo\nbar"));
    }
}
