import rustc::syntax::ast;

export run;

fn run(
    doc: doc::cratedoc,
    crate: @ast::crate
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_fn: fn~(
            f: fold::fold<@ast::crate>,
            d: doc::fndoc
        ) -> doc::fndoc {
            fold_fn(f, d)
        }
        with *fold::default_seq_fold(crate)
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<@ast::crate>,
    doc: doc::fndoc
) -> doc::fndoc {
    import rustc::middle::ast_map;
    import rustc::syntax::print::pprust;

    let crate = fold.ctxt;

    let map = ast_map::map_crate(*crate);

    fn add_ret_ty(
        doc: option<doc::retdoc>,
        tystr: str
    ) -> option<doc::retdoc> {
        alt doc {
          some(doc) {
            fail "unimplemented";
          }
          none. {
            some({
                desc: none,
                ty: some(tystr)
            })
          }
        }
    }

    ~{
        return: alt map.get(doc.id) {
          ast_map::node_item(@{
            node: ast::item_fn(decl, _, _), _
          }) {
            add_ret_ty(doc.return, pprust::ty_to_str(decl.output))
          }
        }
        with *doc
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn should_add_fn_ret_types() {
        let source = "fn a() -> int { }";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "");
        let doc = run(doc, ast);
        assert option::get(doc.topmod.fns[0].return).ty == some("int");
    }
}
