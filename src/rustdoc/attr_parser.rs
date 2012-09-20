#[doc(
    brief = "Attribute parsing",
    desc =
    "The attribute parser provides methods for pulling documentation out of \
     an AST's attributes."
)];

use syntax::ast;
use syntax::attr;
use core::tuple;

export CrateAttrs;
export parse_crate, parse_desc;
export parse_hidden;

type CrateAttrs = {
    name: Option<~str>
};

#[cfg(test)]
mod test {

    fn parse_attributes(source: ~str) -> ~[ast::attribute] {
        use syntax::parse;
        use parse::parser;
        use parse::attr::parser_attr;
        use syntax::codemap;
        use syntax::diagnostic;

        let parse_sess = syntax::parse::new_parse_sess(None);
        let parser = parse::new_parser_from_source_str(
            parse_sess, ~[], ~"-", codemap::fss_none, @source);

        parser.parse_outer_attributes()
    }
}

fn doc_meta(
    attrs: ~[ast::attribute]
) -> Option<@ast::meta_item> {

    /*!
     * Given a vec of attributes, extract the meta_items contained in the \
     * doc attribute
     */

    let doc_metas = doc_metas(attrs);
    if vec::is_not_empty(doc_metas) {
        if vec::len(doc_metas) != 1u {
            warn!("ignoring %u doc attributes", vec::len(doc_metas) - 1u);
        }
        Some(doc_metas[0])
    } else {
        None
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

fn parse_crate(attrs: ~[ast::attribute]) -> CrateAttrs {
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

fn parse_desc(attrs: ~[ast::attribute]) -> Option<~str> {
    match doc_meta(attrs) {
      Some(meta) => {
        attr::get_meta_item_value_str(meta)
      }
      None => None
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

fn parse_hidden(attrs: ~[ast::attribute]) -> bool {
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
