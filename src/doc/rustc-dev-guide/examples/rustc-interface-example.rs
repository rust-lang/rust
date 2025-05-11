// Tested with nightly-2025-03-28

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::registry;
use rustc_hash::FxHashMap;
use rustc_session::config;

fn main() {
    let config = rustc_interface::Config {
        // Command line options
        opts: config::Options::default(),
        // cfg! configuration in addition to the default ones
        crate_cfg: Vec::new(),       // FxHashSet<(String, Option<String>)>
        crate_check_cfg: Vec::new(), // CheckCfg
        input: config::Input::Str {
            name: rustc_span::FileName::Custom("main.rs".into()),
            input: r#"
static HELLO: &str = "Hello, world!";
fn main() {
    println!("{HELLO}");
}
"#
            .into(),
        },
        output_dir: None,  // Option<PathBuf>
        output_file: None, // Option<PathBuf>
        file_loader: None, // Option<Box<dyn FileLoader + Send + Sync>>
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES.to_owned(),
        lint_caps: FxHashMap::default(), // FxHashMap<lint::LintId, lint::Level>
        // This is a callback from the driver that is called when [`ParseSess`] is created.
        psess_created: None, //Option<Box<dyn FnOnce(&mut ParseSess) + Send>>
        // This is a callback from the driver that is called when we're registering lints;
        // it is called during plugin registration when we have the LintStore in a non-shared state.
        //
        // Note that if you find a Some here you probably want to call that function in the new
        // function being registered.
        register_lints: None, // Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>
        // This is a callback from the driver that is called just after we have populated
        // the list of queries.
        //
        // The second parameter is local providers and the third parameter is external providers.
        override_queries: None, // Option<fn(&Session, &mut ty::query::Providers<'_>, &mut ty::query::Providers<'_>)>
        // Registry of diagnostics codes.
        registry: registry::Registry::new(rustc_errors::codes::DIAGNOSTICS),
        make_codegen_backend: None,
        expanded_args: Vec::new(),
        ice_file: None,
        hash_untracked_state: None,
        using_internal_features: &rustc_driver::USING_INTERNAL_FEATURES,
    };
    rustc_interface::run_compiler(config, |compiler| {
        // Parse the program and print the syntax tree.
        let krate = rustc_interface::passes::parse(&compiler.sess);
        println!("{krate:?}");
        // Analyze the program and inspect the types of definitions.
        rustc_interface::create_and_enter_global_ctxt(&compiler, krate, |tcx| {
            for id in tcx.hir_free_items() {
                let item = tcx.hir_item(id);
                match item.kind {
                    rustc_hir::ItemKind::Static(ident, ..)
                    | rustc_hir::ItemKind::Fn { ident, .. } => {
                        let ty = tcx.type_of(item.hir_id().owner.def_id);
                        println!("{ident:?}:\t{ty:?}")
                    }
                    _ => (),
                }
            }
        });
    });
}
