#[doc(
    brief = "Attribute parsing",
    desc =
    "The attribute parser provides methods for pulling documentation out of \
     an AST's attributes."
)];

import rustc::syntax::ast;
import rustc::front::attr;
import core::tuple;

export fn_attrs, arg_attrs;
export parse_fn;

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

fn parse_fn(
    attrs: [ast::attribute]
) -> fn_attrs {

    let no_attrs = {
        brief: none,
        desc: none,
        args: [],
        return: none
    };

    ret alt doc_meta(attrs) {
      some(meta) {
        alt attr::get_meta_item_value_str(meta) {
          some(desc) {
            {
                brief: none,
                desc: some(desc),
                args: [],
                return: none
            }
          }
          none. {
            alt attr::get_meta_item_list(meta) {
              some(list) {
                parse_fn_(list)
              }
              none. {
                no_attrs
              }
            }
          }
        }
      }
      none. {
        no_attrs
      }
    };
}

fn parse_fn_(
    items: [@ast::meta_item]
) -> fn_attrs {
    let brief = attr::meta_item_value_from_list(items, "brief");
    let desc = attr::meta_item_value_from_list(items, "desc");
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
      none. { [] }
    };

    {
        brief: brief,
        desc: desc,
        args: args,
        return: return
    }
}

#[cfg(test)]
mod tests {

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

    #[test]
    fn parse_fn_should_handle_undocumented_functions() {
        let source = "";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.brief == none;
        assert attrs.desc == none;
        assert attrs.return == none;
        assert vec::len(attrs.args) == 0u;
    }

    #[test]
    fn parse_fn_should_parse_simple_doc_attributes() {
        let source = "#[doc = \"basic\"]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.desc == some("basic");
    }

    #[test]
    fn parse_fn_should_parse_the_brief_description() {
        let source = "#[doc(brief = \"short\")]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.brief == some("short");
    }

    #[test]
    fn parse_fn_should_parse_the_long_description() {
        let source = "#[doc(desc = \"description\")]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.desc == some("description");
    }

    #[test]
    fn parse_fn_should_parse_the_return_value_description() {
        let source = "#[doc(return = \"return value\")]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.return == some("return value");
    }

    #[test]
    fn parse_fn_should_parse_the_argument_descriptions() {
        let source = "#[doc(args(a = \"arg a\", b = \"arg b\"))]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.args[0] == {name: "a", desc: "arg a"};
        assert attrs.args[1] == {name: "b", desc: "arg b"};
    }
}
