#[doc(
    brief = "Attribute parsing",
    desc =
    "The attribute parser provides methods for pulling documentation out of \
     an AST's attributes."
)];

import rustc::syntax::ast;
import rustc::front::attr;
import core::tuple;

export crate_attrs, mod_attrs, fn_attrs, arg_attrs;
export parse_crate, parse_mod, parse_fn;

type crate_attrs = {
    name: option<str>,
    desc: option<str>
};

type mod_attrs = {
    brief: option<str>,
    desc: option<str>
};

type fn_attrs = {
    brief: option<str>,
    desc: option<str>,
    args: [arg_attrs],
    return: option<str>
};

type arg_attrs = {
    name: str,
    desc: str
};

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
    let attr_metas = attr::attr_metas(attrs);

    {
        name: attr::meta_item_value_from_list(link_metas, "name"),
        desc: attr::meta_item_value_from_list(attr_metas, "desc")
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

#[test]
fn should_extract_crate_desc() {
    let source = "#[desc = \"Teddybears\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_crate(attrs);
    assert attrs.desc == some("Teddybears");
}

fn parse_mod(attrs: [ast::attribute]) -> mod_attrs {
    parse_short_doc_or(
        attrs,
        {|desc|
            {
                brief: none,
                desc: desc
            }
        },
        parse_mod_long_doc
    )
}

fn parse_mod_long_doc(
    _items: [@ast::meta_item],
    brief: option<str>,
    desc: option<str>
) -> mod_attrs {
    {
        brief: brief,
        desc: desc
    }
}

#[test]
fn parse_mod_should_handle_undocumented_mods() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_mod(attrs);
    assert attrs.brief == none;
    assert attrs.desc == none;
}

#[test]
fn parse_mod_should_parse_simple_doc_attributes() {
    let source = "#[doc = \"basic\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_mod(attrs);
    assert attrs.desc == some("basic");
}

#[test]
fn parse_mod_should_parse_the_brief_description() {
    let source = "#[doc(brief = \"short\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_mod(attrs);
    assert attrs.brief == some("short");
}

#[test]
fn parse_mod_should_parse_the_long_description() {
    let source = "#[doc(desc = \"description\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_mod(attrs);
    assert attrs.desc == some("description");
}

fn parse_short_doc_or<T>(
    attrs: [ast::attribute],
    handle_short: fn&(
        short_desc: option<str>
    ) -> T,
    parse_long: fn&(
        doc_items: [@ast::meta_item],
        brief: option<str>,
        desc: option<str>
    ) -> T
) -> T {
    alt doc_meta(attrs) {
      some(meta) {
        alt attr::get_meta_item_value_str(meta) {
          some(desc) { handle_short(some(desc)) }
          none {
            alt attr::get_meta_item_list(meta) {
              some(list) {
                let brief = attr::meta_item_value_from_list(list, "brief");
                let desc = attr::meta_item_value_from_list(list, "desc");
                parse_long(list, brief, desc)
              }
              none {
                handle_short(none)
              }
            }
          }
        }
      }
      none {
        handle_short(none)
      }
    }
}

fn parse_fn(
    attrs: [ast::attribute]
) -> fn_attrs {

    parse_short_doc_or(
        attrs,
        {|desc|
            {
                brief: none,
                desc: desc,
                args: [],
                return: none
            }
        },
        parse_fn_long_doc
    )
}

fn parse_fn_long_doc(
    items: [@ast::meta_item],
    brief: option<str>,
    desc: option<str>
) -> fn_attrs {
    let return = attr::meta_item_value_from_list(items, "return");

    let args = alt attr::meta_item_list_from_list(items, "args") {
      some(items) {
        vec::filter_map(items) {|item|
            option::map(attr::name_value_str_pair(item)) { |pair|
                {
                    name: tuple::first(pair),
                    desc: tuple::second(pair)
                }
            }
        }
      }
      none { [] }
    };

    {
        brief: brief,
        desc: desc,
        args: args,
        return: return
    }
}

#[test]
fn parse_fn_should_handle_undocumented_functions() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.brief == none;
    assert attrs.desc == none;
    assert attrs.return == none;
    assert vec::len(attrs.args) == 0u;
}

#[test]
fn parse_fn_should_parse_simple_doc_attributes() {
    let source = "#[doc = \"basic\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.desc == some("basic");
}

#[test]
fn parse_fn_should_parse_the_brief_description() {
    let source = "#[doc(brief = \"short\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.brief == some("short");
}

#[test]
fn parse_fn_should_parse_the_long_description() {
    let source = "#[doc(desc = \"description\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.desc == some("description");
}

#[test]
fn parse_fn_should_parse_the_return_value_description() {
    let source = "#[doc(return = \"return value\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.return == some("return value");
}

#[test]
fn parse_fn_should_parse_the_argument_descriptions() {
    let source = "#[doc(args(a = \"arg a\", b = \"arg b\"))]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.args[0] == {name: "a", desc: "arg a"};
    assert attrs.args[1] == {name: "b", desc: "arg b"};
}

#[cfg(test)]
mod test {

    fn parse_attributes(source: str) -> [ast::attribute] {
        import rustc::syntax::parse::parser;
        // FIXME: Uncommenting this results in rustc bugs
        //import rustc::syntax::codemap;
        import rustc::driver::diagnostic;

        let cm = rustc::syntax::codemap::new_codemap();
        let parse_sess = @{
            cm: cm,
            mutable next_id: 0,
            diagnostic: diagnostic::mk_handler(cm, none)
        };
        let parser = parser::new_parser_from_source_str(
            parse_sess, [], "-", source);

        parser::parse_outer_attributes(parser)
    }
}
