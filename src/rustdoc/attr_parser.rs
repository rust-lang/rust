#[doc(
    brief = "Attribute parsing",
    desc =
    "The attribute parser provides methods for pulling documentation out of \
     an AST's attributes."
)];

import rustc::syntax::ast;
import rustc::front::attr;
import core::tuple;

export crate_attrs, basic_attrs, fn_attrs, arg_attrs,
       variant_attrs, res_attrs, method_attrs;
export parse_crate, parse_basic, parse_fn,
       parse_variant, parse_res, parse_method;
export parse_hidden;

type crate_attrs = {
    name: option<str>
};

type basic_attrs = {
    brief: option<str>,
    desc: option<str>
};

type fn_attrs = {
    args: [arg_attrs],
    return: option<str>,
    failure: option<str>
};

type arg_attrs = {
    name: str,
    desc: str
};

type variant_attrs = {
    desc: option<str>
};

type res_attrs = {
    args: [arg_attrs]
};

type method_attrs = fn_attrs;

#[cfg(test)]
mod test {

    fn parse_attributes(source: str) -> [ast::attribute] {
        import rustc::syntax::parse::parser;
        import rustc::syntax::codemap;
        import rustc::driver::diagnostic;

        let cm = codemap::new_codemap();
        let handler = diagnostic::mk_handler(none);
        let parse_sess = @{
            cm: cm,
            mutable next_id: 0,
            span_diagnostic: diagnostic::mk_span_handler(handler, cm),
            mutable chpos: 0u,
            mutable byte_pos: 0u
        };
        let parser = parser::new_parser_from_source_str(
            parse_sess, [], "-", codemap::fss_none, @source);

        parser::parse_outer_attributes(parser)
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
        name: attr::meta_item_value_from_list(link_metas, "name")
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

fn parse_basic(
    attrs: [ast::attribute]
) -> {
    brief: option<str>,
    desc: option<str>
} {
    parse_short_doc_or(
        attrs,
        {|desc|
            {
                brief: none,
                desc: desc
            }
        },
        {|_items, brief, desc|
            {
                brief: brief,
                desc: desc
            }
        }
    )
}

#[test]
fn parse_basic_should_handle_undocumented_mods() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_basic(attrs);
    assert attrs.brief == none;
    assert attrs.desc == none;
}

#[test]
fn parse_basic_should_parse_simple_doc_attributes() {
    let source = "#[doc = \"basic\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_basic(attrs);
    assert attrs.desc == some("basic");
}

#[test]
fn parse_basic_should_parse_the_brief_description() {
    let source = "#[doc(brief = \"short\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_basic(attrs);
    assert attrs.brief == some("short");
}

#[test]
fn parse_basic_should_parse_the_long_description() {
    let source = "#[doc(desc = \"description\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_basic(attrs);
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

fn parse_long_doc<T>(
    attrs: [ast::attribute],
    parse_long: fn&(doc_items: [@ast::meta_item]) -> T
) -> T {
    alt doc_meta(attrs) {
      some(meta) {
        alt attr::get_meta_item_list(meta) {
          some(list) {
            parse_long(list)
          }
          none {
            parse_long([])
          }
        }
      }
      none { parse_long([]) }
    }
}

fn parse_fn(attrs: [ast::attribute]) -> fn_attrs {
    parse_long_doc(attrs, parse_fn_long_doc)
}

fn parse_fn_long_doc(items: [@ast::meta_item]) -> fn_attrs {
    let return = attr::meta_item_value_from_list(items, "return");
    let failure = attr::meta_item_value_from_list(items, "failure");
    let args = parse_args(items);

    {
        args: args,
        return: return,
        failure: failure
    }
}

fn parse_args(items: [@ast::meta_item]) -> [arg_attrs] {
    alt attr::meta_item_list_from_list(items, "args") {
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
    }
}

#[test]
fn parse_fn_should_handle_undocumented_functions() {
    let source = "";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.return == none;
    assert vec::len(attrs.args) == 0u;
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

#[test]
fn parse_fn_should_parse_failure_conditions() {
    let source = "#[doc(failure = \"it's the fail\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_fn(attrs);
    assert attrs.failure == some("it's the fail");
}

fn parse_variant(attrs: [ast::attribute]) -> variant_attrs {
    parse_short_doc_or(
        attrs,
        {|desc|
            {
                desc: desc
            }
        },
        {|_items, brief, desc|
            if option::is_some(brief) && option::is_some(desc) {
                // FIXME: Warn about dropping brief description
            }

            {
                // Prefer desc over brief
                desc: option::maybe(brief, desc, {|s| some(s) })
            }
        }
    )
}

#[test]
fn should_parse_variant_short_doc() {
    let source = "#[doc = \"a\"]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_variant(attrs);
    assert attrs.desc == some("a");
}

#[test]
fn should_parse_variant_brief_doc() {
    let source = "#[doc(brief = \"a\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_variant(attrs);
    assert attrs.desc == some("a");
}

#[test]
fn should_parse_variant_long_doc() {
    let source = "#[doc(desc = \"a\")]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_variant(attrs);
    assert attrs.desc == some("a");
}

fn parse_res(attrs: [ast::attribute]) -> res_attrs {
    parse_long_doc(attrs, parse_res_long_doc)
}

fn parse_res_long_doc(items: [@ast::meta_item]) -> res_attrs {
    {
        args: parse_args(items)
    }
}

#[test]
fn shoulde_parse_resource_arg() {
    let source = "#[doc(args(a = \"b\"))]";
    let attrs = test::parse_attributes(source);
    let attrs = parse_res(attrs);
    assert attrs.args[0].name == "a";
    assert attrs.args[0].desc == "b";
}

fn parse_method(attrs: [ast::attribute]) -> method_attrs {
    parse_fn(attrs)
}

fn parse_hidden(attrs: [ast::attribute]) -> bool {
    parse_short_doc_or(
        attrs,
        {|_desc| false },
        {|metas, _brief, _desc|
            let hiddens = attr::find_meta_items_by_name(metas, "hidden");
            vec::is_not_empty(hiddens)
        }
    )
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
