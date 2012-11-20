/*!
Attribute parsing

The attribute parser provides methods for pulling documentation out of
an AST's attributes.
*/

use syntax::ast;
use syntax::attr;
use core::tuple;

pub type CrateAttrs = {
    name: Option<~str>
};

#[cfg(test)]
mod test {
    #[legacy_exports];

    fn parse_attributes(source: ~str) -> ~[ast::attribute] {
        use syntax::parse;
        use parse::parser;
        use parse::attr::parser_attr;
        use syntax::codemap;
        use syntax::diagnostic;

        let parse_sess = syntax::parse::new_parse_sess(None);
        let parser = parse::new_parser_from_source_str(
            parse_sess, ~[], ~"-", codemap::FssNone, @source);

        parser.parse_outer_attributes()
    }
}

fn doc_metas(
    attrs: ~[ast::attribute]
) -> ~[@ast::meta_item] {

    let doc_attrs = attr::find_attrs_by_name(attrs, ~"doc");
    let doc_metas = do doc_attrs.map |attr| {
        attr::attr_meta(attr::desugar_doc_attr(attr))
    };

    return doc_metas;
}

pub fn parse_crate(attrs: ~[ast::attribute]) -> CrateAttrs {
    let link_metas = attr::find_linkage_metas(attrs);

    {
        name: attr::last_meta_item_value_str_by_name(link_metas, ~"name")
    }
}

#[test]
fn should_extract_crate_name_from_link_attribute() {
    let source = ~"#[link(name = \"snuggles\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == Some(~"snuggles");
}

#[test]
fn should_not_extract_crate_name_if_no_link_attribute() {
    let source = ~"";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == None;
}

#[test]
fn should_not_extract_crate_name_if_no_name_value_in_link_attribute() {
    let source = ~"#[link(whatever)]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == None;
}

pub fn parse_desc(attrs: ~[ast::attribute]) -> Option<~str> {
    let doc_strs = do doc_metas(attrs).filter_map |meta| {
        attr::get_meta_item_value_str(*meta)
    };
    if doc_strs.is_empty() {
        None
    } else {
        Some(str::connect(doc_strs, "\n"))
    }
}

#[test]
fn parse_desc_should_handle_undocumented_mods() {
    let source = ~"";
    let attrs = test::parse_attributes(source);
    let attrs = parse_desc(attrs);
    assert attrs == None;
}

#[test]
fn parse_desc_should_parse_simple_doc_attributes() {
    let source = ~"#[doc = \"basic\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_desc(attrs);
    assert attrs == Some(~"basic");
}

pub fn parse_hidden(attrs: ~[ast::attribute]) -> bool {
    do doc_metas(attrs).find |meta| {
        match attr::get_meta_item_list(meta) {
          Some(metas) => {
            let hiddens = attr::find_meta_items_by_name(metas, ~"hidden");
            vec::is_not_empty(hiddens)
          }
          None => false
        }
    }.is_some()
}

#[test]
fn should_parse_hidden_attribute() {
    let source = ~"#[doc(hidden)]";
    let attrs = test::parse_attributes(source);
    assert parse_hidden(attrs) == true;
}

#[test]
fn should_parse_hidden_attribute_with_other_docs() {
    let source = ~"#[doc = \"foo\"] #[doc(hidden)] #[doc = \"foo\"]";
    let attrs = test::parse_attributes(source);
    assert parse_hidden(attrs) == true;
}

#[test]
fn should_not_parse_non_hidden_attribute() {
    let source = ~"#[doc = \"\"]";
    let attrs = test::parse_attributes(source);
    assert parse_hidden(attrs) == false;
}

#[test]
fn should_concatenate_multiple_doc_comments() {
    let source = ~"/// foo\n/// bar";
    let desc = parse_desc(test::parse_attributes(source));
    assert desc == Some(~"foo\nbar");
}


