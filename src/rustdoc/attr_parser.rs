export parse_fn;

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
fn parse_fn(items: [@ast::meta_item]) -> doc::fndoc {
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
