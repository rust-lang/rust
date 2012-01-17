import rustc::syntax::ast;
import rustc::middle::ast_map;

export run;

fn run(
    srv: astsrv::seq_srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_fn: fn~(
            f: fold::fold<astsrv::seq_srv>,
            d: doc::fndoc
        ) -> doc::fndoc {
            fold_fn(f, d)
        }
        with *fold::default_seq_fold(srv)
    });
    fold.fold_crate(fold, doc)
}

fn fold_fn(
    fold: fold::fold<astsrv::seq_srv>,
    doc: doc::fndoc
) -> doc::fndoc {

    let srv = fold.ctxt;

    let attrs = alt srv.map.get(doc.id) {
      ast_map::node_item(item) { item.attrs }
    };
    let attrs = attr_parser::parse_fn(attrs);
    ret merge_fn_attrs(doc, attrs);

    fn merge_fn_attrs(
        doc: doc::fndoc,
        attrs: attr_parser::fn_attrs
    ) -> doc::fndoc {
        ret ~{
            id: doc.id,
            name: doc.name,
            brief: attrs.brief,
            desc: attrs.desc,
            return: merge_ret_attrs(doc.return, attrs.return),
            args: merge_arg_attrs(doc.args, attrs.args)
        };
    }

    fn merge_arg_attrs(
        doc: [(str, str)],
        _attrs: [attr_parser::arg_attrs]
    ) -> [(str, str)] {
        // FIXME
        doc
    }

    fn merge_ret_attrs(
        doc: option<doc::retdoc>,
        _attrs: option<str>
    ) -> option<doc::retdoc> {
        // FIXME
        doc
    }
}

#[test]
fn fold_fn_should_extract_fn_attributes() {
    let source = "#[doc = \"test\"] fn a() -> int { }";
    let srv = astsrv::mk_seq_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let fold = fold::default_seq_fold(srv);
    let doc = fold_fn(fold, doc.topmod.fns[0]);
    assert doc.desc == some("test");
}
