import rustc::syntax::ast;

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

fn parse_fn(
    attrs: [ast::attribute]
) -> fn_attrs {

    for attr in attrs {
        alt attr.node.value.node {
          ast::meta_name_value(
              "doc", {node: ast::lit_str(value), span: _}) {
            ret {
                brief: none,
                desc: some(value),
                args: [],
                return: none
            };
          }
          ast::meta_list("doc", docs) {
            ret parse_fn_(docs);
          }
          _ { }
        }
    }

    {
        brief: none,
        desc: none,
        args: [],
        return: none
    }
}

fn parse_fn_(
    items: [@ast::meta_item]
) -> fn_attrs {
    let brief = none;
    let desc = none;
    let return = none;
    let argdocs = [];
    let argdocsfound = none;
    for item: @ast::meta_item in items {
        alt item.node {
            ast::meta_name_value("brief", {node: ast::lit_str(value),
                                           span: _}) {
                brief = some(value);
            }
            ast::meta_name_value("desc", {node: ast::lit_str(value),
                                              span: _}) {
                desc = some(value);
            }
            ast::meta_name_value("return", {node: ast::lit_str(value),
                                            span: _}) {
                return = some(value);
            }
            ast::meta_list("args", args) {
                argdocsfound = some(args);
            }
            _ { }
        }
    }

    alt argdocsfound {
        none. { }
        some(ds) {
            for d: @ast::meta_item in ds {
                alt d.node {
                  ast::meta_name_value(key, {node: ast::lit_str(value),
                                             span: _}) {
                    argdocs += [{
                        name: key,
                        desc: value
                    }];
                  }
                }
            }
        }
    }

    {
        brief: brief,
        desc: desc,
        args: argdocs,
        return: return
    }
}

#[cfg(test)]
mod tests {

    fn parse_attributes(source: str) -> [ast::attribute] {
        import rustc::driver::diagnostic;
        import rustc::syntax::codemap;
        import rustc::syntax::parse::parser;

        let cm = codemap::new_codemap();
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

    #[tes]
    fn parse_fn_should_parse_simple_doc_attributes() {
        let source = "#[doc = \"basic\"]";
        let attrs = parse_attributes(source);
        let attrs = parse_fn(attrs);
        assert attrs.brief == some("basic");
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