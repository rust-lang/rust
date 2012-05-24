#[doc(
    brief = "Attribute parsing",
    desc =
    "The attribute parser provides methods for pulling documentation out of \
     an AST's attributes."
)];

import syntax::ast;
import syntax::attr;
import core::tuple;

export crate_attrs;
export parse_crate, parse_desc;
export parse_hidden;

type crate_attrs = {
    name: option<str>
};

#[cfg(test)]
mod test {

    fn parse_attributes(source: str) -> [ast::attribute] {
        import syntax::parse;
        import parse::parser;
        import parse::attr::parser_attr;
        import syntax::codemap;
        import syntax::diagnostic;

        let cm = codemap::new_codemap();
        let handler = diagnostic::mk_handler(none);
        let parse_sess = @{
            cm: cm,
            mut next_id: 0,
            span_diagnostic: diagnostic::mk_span_handler(handler, cm),
            mut chpos: 0u,
            mut byte_pos: 0u
        };
        let parser = parse::new_parser_from_source_str(
            parse_sess, [], "-", codemap::fss_none, @source);

        parser.parse_outer_attributes()
    }
}

fn doc_meta(
    attrs: [ast::attribute]
) -> option<@ast::meta_item> {

    #[doc =
      "Given a vec of attributes, extract the meta_items contained in the \
       doc attribute"];

    let doc_attrs = attr::find_attrs_by_name(attrs, "doc");
    let doc_metas = attr::attr_metas(doc_attrs);
    if vec::is_not_empty(doc_metas) {
        if vec::len(doc_metas) != 1u {
            #warn("ignoring %u doc attributes", vec::len(doc_metas) - 1u);
        }
        some(doc_metas[0])
    } else {
        none
    }
}

fn parse_crate(attrs: [ast::attribute]) -> crate_attrs {
    let link_metas = attr::find_linkage_metas(attrs);

    {
        name: attr::last_meta_item_value_str_by_name(link_metas, "name")
    }
}

#[test]
fn should_extract_crate_name_from_link_attribute() {
    let source = "#[link(name = \"snuggles\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == some("snuggles");
}

#[test]
fn should_not_extract_crate_name_if_no_link_attribute() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == none;
}

#[test]
fn should_not_extract_crate_name_if_no_name_value_in_link_attribute() {
    let source = "#[link(whatever)]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.name == none;
}

fn parse_desc(attrs: [ast::attribute]) -> option<str> {
    alt doc_meta(attrs) {
      some(meta) {
        attr::get_meta_item_value_str(meta)
      }
      none { none }
    }
}

#[test]
fn parse_desc_should_handle_undocumented_mods() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_desc(attrs);
    assert attrs == none;
}

#[test]
fn parse_desc_should_parse_simple_doc_attributes() {
    let source = "#[doc = \"basic\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_desc(attrs);
    assert attrs == some("basic");
}

fn parse_hidden(attrs: [ast::attribute]) -> bool {
    alt doc_meta(attrs) {
      some(meta) {
        alt attr::get_meta_item_list(meta) {
          some(metas) {
            let hiddens = attr::find_meta_items_by_name(metas, "hidden");
            vec::is_not_empty(hiddens)
          }
          none { false }
        }
      }
      none { false }
    }
}

#[test]
fn shoulde_parse_hidden_attribute() {
    let source = "#[doc(hidden)]";
    let attrs = test::parse_attributes(source);
    assert parse_hidden(attrs) == true;
}

#[test]
fn shoulde_not_parse_non_hidden_attribute() {
    let source = "#[doc = \"\"]";
    let attrs = test::parse_attributes(source);
    assert parse_hidden(attrs) == false;
}
