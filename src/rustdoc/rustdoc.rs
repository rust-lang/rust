// Some utility interfaces
import doc::item;
import doc::util;

/// A single operation on the document model
type pass = {
    name: str,
    f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc
};

fn run_passes(
    srv: astsrv::srv,
    doc: doc::doc,
    passes: ~[pass]
) -> doc::doc {

    #[doc(
        brief =
        "Run a series of passes over the document",
        args(
            srv =
            "The AST service to provide to the passes",
            doc =
            "The document to transform",
            passes =
            "The list of passes used to transform the document"
        ),
        return =
        "The transformed document that results from folding the \
         original through each pass"
    )];

    let mut passno = 0;
    do vec::foldl(doc, passes) |doc, pass| {
        log(debug, #fmt("pass #%d", passno));
        passno += 1;
        log(debug, doc);
        do time(pass.name) {
            pass.f(srv, doc)
        }
    }
}

#[test]
fn test_run_passes() {
    fn pass1(
        _srv: astsrv::srv,
        doc: doc::doc
    ) -> doc::doc {
        {
            pages: ~[
                doc::cratepage({
                    topmod: {
                        item: {
                            name: doc.cratemod().name() + "two"
                            with doc.cratemod().item
                        },
                        items: ~[],
                        index: none
                    }
                })
            ]
        }
    }
    fn pass2(
        _srv: astsrv::srv,
        doc: doc::doc
    ) -> doc::doc {
        {
            pages: ~[
                doc::cratepage({
                    topmod: {
                        item: {
                            name: doc.cratemod().name() + "three"
                            with doc.cratemod().item
                        },
                        items: ~[],
                        index: none
                    }
                })
            ]
        }
    }
    let source = "";
    do astsrv::from_str(source) |srv| {
        let passes = ~[
            {
                name: "",
                f: pass1
            },
            {
                name: "",
                f: pass2
            }
        ];
        let doc = extract::from_srv(srv, "one");
        let doc = run_passes(srv, doc, passes);
        assert doc.cratemod().name() == "onetwothree";
    }
}

fn main(args: ~[str]) {

    if vec::contains(args, "-h") {
        config::usage();
        ret;
    }

    let config = alt config::parse_config(args) {
      result::ok(config) { config }
      result::err(err) {
        io::println(#fmt("error: %s", err));
        ret;
      }
    };

    run(config);
}

fn time<T>(what: str, f: fn() -> T) -> T {
    let start = std::time::precise_time_s();
    let rv = f();
    let end = std::time::precise_time_s();
    #info("time: %3.3f s    %s", end - start, what);
    ret rv;
}

/// Runs rustdoc over the given file
fn run(config: config::config) {

    let source_file = config.input_crate;
    do astsrv::from_file(source_file) |srv| {
        do time("wait_ast") {
            do astsrv::exec(srv) |_ctxt| { }
        };
        let doc = time("extract", || {
            let default_name = source_file;
            extract::from_srv(srv, default_name)
        });
        run_passes(srv, doc, ~[
            reexport_pass::mk_pass(),
            prune_unexported_pass::mk_pass(),
            tystr_pass::mk_pass(),
            path_pass::mk_pass(),
            attr_pass::mk_pass(),
            escape_pass::mk_pass(),
            prune_hidden_pass::mk_pass(),
            desc_to_brief_pass::mk_pass(),
            unindent_pass::mk_pass(),
            sectionalize_pass::mk_pass(),
            trim_pass::mk_pass(),
            sort_item_name_pass::mk_pass(),
            sort_item_type_pass::mk_pass(),
            markdown_index_pass::mk_pass(config),
            page_pass::mk_pass(config.output_style),
            markdown_pass::mk_pass(
                markdown_writer::make_writer_factory(config)
            )
        ]);
    }
}
