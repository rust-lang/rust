use doc::ItemUtils;
use doc::Item;
use pass::Pass;
use config::Config;

fn main(args: ~[~str]) {

    if args.contains(~"-h") || args.contains(~"--help") {
        config::usage();
        return;
    }

    let config = match config::parse_config(args) {
      Ok(config) => config,
      Err(err) => {
        io::println(fmt!("error: %s", err));
        return;
      }
    };

    run(config);
}

/// Runs rustdoc over the given file
fn run(config: Config) {

    let source_file = config.input_crate;

    // Create an AST service from the source code
    do astsrv::from_file(source_file.to_str()) |srv| {

        // Just time how long it takes for the AST to become available
        do time(~"wait_ast") {
            do astsrv::exec(srv) |_ctxt| { }
        };

        // Extract the initial doc tree from the AST. This contains
        // just names and node ids.
        let doc = time(~"extract", || {
            let default_name = source_file;
            extract::from_srv(srv, default_name.to_str())
        });

        // Refine and publish the document
        pass::run_passes(srv, doc, ~[
            // Generate type and signature strings
            tystr_pass::mk_pass(),
            // Record the full paths to various nodes
            path_pass::mk_pass(),
            // Extract the docs attributes and attach them to doc nodes
            attr_pass::mk_pass(),
            // Perform various text escaping
            escape_pass::mk_pass(),
            // Remove things marked doc(hidden)
            prune_hidden_pass::mk_pass(),
            // Remove things that are private
            // XXX enable this after 'export' is removed in favor of 'pub'
            // prune_private_pass::mk_pass(),
            // Extract brief documentation from the full descriptions
            desc_to_brief_pass::mk_pass(),
            // Massage the text to remove extra indentation
            unindent_pass::mk_pass(),
            // Split text into multiple sections according to headers
            sectionalize_pass::mk_pass(),
            // Trim extra spaces from text
            trim_pass::mk_pass(),
            // Sort items by name
            sort_item_name_pass::mk_pass(),
            // Sort items again by kind
            sort_item_type_pass::mk_pass(),
            // Create indexes appropriate for markdown
            markdown_index_pass::mk_pass(config),
            // Break the document into pages if required by the
            // output format
            page_pass::mk_pass(config.output_style),
            // Render
            markdown_pass::mk_pass(
                markdown_writer::make_writer_factory(config)
            )
        ]);
    }
}

fn time<T>(what: ~str, f: fn() -> T) -> T {
    let start = std::time::precise_time_s();
    let rv = f();
    let end = std::time::precise_time_s();
    info!("time: %3.3f s    %s", end - start, what);
    return rv;
}
