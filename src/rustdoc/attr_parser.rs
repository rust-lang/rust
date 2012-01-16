export parse_fn;

fn parse_fn(attrs: [ast::attribute]) -> doc::fndoc {
    let noargdocs = map::new_str_hash::<str>();
    let _fndoc = none;
    for attr: ast::attribute in attrs {
        alt attr.node.value.node {
            ast::meta_name_value(
                "doc", {node: ast::lit_str(value), span: _}) {
                _fndoc = some(~{
                    name: "todo",
                    brief: value,
                    desc: none,
                    return: none,
                    args: noargdocs
                });
            }
            ast::meta_list("doc", docs) {
                _fndoc = some(
                    parse_fn_(docs));
            }
        }
    }

    let _fndoc0 = alt _fndoc {
        some(_d) { _d }
        none. {
          ~{
              name: "todo",
              brief: "_undocumented_",
              desc: none,
              return: none,
              args: noargdocs
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
fn parse_fn_(items: [@ast::meta_item]) -> doc::fndoc {
    let brief = none;
    let desc = none;
    let return = none;
    let argdocs = map::new_str_hash::<str>();
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
                        argdocs.insert(key, value);
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
        name: "todo",
        brief: _brief,
        desc: desc,
        return: return,
        args: argdocs }
}
