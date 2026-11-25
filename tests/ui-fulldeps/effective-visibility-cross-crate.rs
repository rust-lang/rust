//@ run-pass
//@ check-run-results
//@ run-flags: --sysroot {{sysroot-base}}
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_span;

use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_metadata::creader::CStore;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, DefIndex};

const AUX: &str = r#"
pub mod m {
    pub struct Direct;
}
mod private {
    pub struct Reexported;
    pub struct Reachable;
    pub struct ThroughImplTrait;
    pub struct Hidden;
}
pub use private::Reexported;
pub fn reachable() -> private::Reachable { private::Reachable }
pub fn opaque() -> impl Sized { private::ThroughImplTrait }
"#;

struct Aux;
impl Callbacks for Aux {}

struct Main;
impl Callbacks for Main {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let krate = tcx.crates(()).iter().copied().find(|&c| tcx.crate_name(c).as_str() == "aux");
        let krate = krate.unwrap();
        for index in 0..CStore::from_tcx(tcx).num_def_ids_untracked(krate) {
            let def_id = DefId { krate, index: DefIndex::from_usize(index) };
            if let Some(level) = tcx.effective_visibility_level(def_id) {
                println!("{}: {level:?}", tcx.def_path(def_id).to_string_no_crate_verbose());
            }
        }
        Compilation::Stop
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let run = |extra: &[&str], callbacks: &mut (dyn Callbacks + Send)| {
        let mut args = args.clone();
        args.extend(extra.iter().map(|s| s.to_string()));
        rustc_driver::catch_fatal_errors(|| rustc_driver::compiler_entrypoint(&args, callbacks))
            .unwrap();
    };

    std::fs::write("aux.rs", AUX).unwrap();
    run(&["--crate-type=lib", "--emit=metadata", "-Awarnings", "aux.rs"], &mut Aux);
    std::fs::write("main.rs", "extern crate aux;").unwrap();
    run(&["--crate-type=lib", "--extern=aux=libaux.rmeta", "main.rs"], &mut Main);
}
