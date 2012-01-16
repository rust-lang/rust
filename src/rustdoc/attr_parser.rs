import rustc::syntax::ast;

export parse_fn;

fn parse_fn(
    name: str,
    id: ast::node_id,
    attrs: [ast::attribute]
) -> doc::fndoc {
    let _fndoc = none;
    for attr: ast::attribute in attrs {
        alt attr.node.value.node {
            ast::meta_name_value(
                "doc", {node: ast::lit_str(value), span: _}) {
                _fndoc = some(~{
                    id: id,
                    name: name,
                    brief: value,
                    desc: none,
                    return: none,
                    args: []
                });
            }
            ast::meta_list("doc", docs) {
                _fndoc = some(
                    parse_fn_(name, id, docs));
            }
        }
    }

    let _fndoc0 = alt _fndoc {
        some(_d) { _d }
        none. {
          ~{
              id: id,
              name: name,
              brief: "_undocumented_",
              desc: none,
              return: none,
              args: []
          }
        }
    };

    ret _fndoc0;
}

#[doc(
  brief = "Parses function docs from a complex #[doc] attribute.",
  desc = "Supported attributes:

* `brief`: Brief description
* `desc`: Long description
* `return`: Description of return value
* `args`: List of argname = argdesc pairs
",
  args(items = "Doc attribute contents"),
  return = "Parsed function docs."
)]
fn parse_fn_(
    name: str,
    id: ast::node_id,
    items: [@ast::meta_item]
) -> doc::fndoc {
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
                    argdocs += [(key, value)];
                  }
                }
            }
        }
    }

    let _brief = alt brief {
        some(_b) { _b }
        none. { "_undocumented_" }
    };

    ~{
        id: id,
        name: name,
        brief: _brief,
        desc: desc,
        return: return,
        args: argdocs }
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
        let doc = parse_fn("f", 0, attrs);
        assert doc.brief == "_undocumented_";
        assert doc.desc == none;
        assert doc.return == none;
        assert vec::len(doc.args) == 0u;
    }

    #[test]
    fn parse_fn_should_parse_simple_doc_attributes() {
        let source = "#[doc = \"basic\"]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.brief == "basic";
    }

    #[test]
    fn parse_fn_should_parse_the_brief_description() {
        let source = "#[doc(brief = \"short\")]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.brief == "short";
    }

    #[test]
    fn parse_fn_should_parse_the_long_description() {
        let source = "#[doc(desc = \"description\")]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.desc == some("description");
    }

    #[test]
    fn parse_fn_should_parse_the_return_value_description() {
        let source = "#[doc(return = \"return value\")]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.return == some("return value");
    }

    #[test]
    fn parse_fn_should_parse_the_argument_descriptions() {
        let source = "#[doc(args(a = \"arg a\", b = \"arg b\"))]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.args[0] == ("a", "arg a");
        assert doc.args[1] == ("b", "arg b");
    }

    #[test]
    fn parse_fn_should_set_brief_desc_to_undocumented_if_not_exists() {
        let source = "#[doc(desc = \"long desc\")]";
        let attrs = parse_attributes(source);
        let doc = parse_fn("f", 0, attrs);
        assert doc.brief == "_undocumented_";
    }
}