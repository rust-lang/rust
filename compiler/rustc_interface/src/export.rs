use std::fs::File;
use std::io::{self, BufWriter, Write};

use rustc_ast_pretty::pprust::{self, AnnNode, PpAnn, State};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{OutFileName, OutputType};

struct InterfaceAnn;

impl PpAnn for InterfaceAnn {
    fn nested(&self, state: &mut State<'_>, node: AnnNode<'_>) {
        // Insert empty fn bodies.
        if let AnnNode::Block(_) = node {
            state.nbsp();
            state.word_nbsp("loop {}");
            return;
        }

        pprust::state::print_default_nested_ann(state, node);
    }
}

pub fn write_interface<'tcx>(tcx: TyCtxt<'tcx>) {
    let sess = tcx.sess;
    if !sess.opts.output_types.contains_key(&OutputType::Interface) {
        return;
    }
    let _timer = sess.timer("write_interface");

    let krate = &tcx.resolver_for_lowering().borrow().1;
    let krate = pprust::state::print_crate_with_erased_comments(
        &krate,
        &InterfaceAnn,
        true,
        sess.edition(),
        &sess.psess.attr_id_generator,
    );
    let outputs = tcx.output_filenames(());
    let export_output = outputs.path(OutputType::Interface);
    match export_output {
        OutFileName::Stdout => {
            let mut file = BufWriter::new(io::stdout());
            let _ = write!(file, "{}", krate);
        }
        OutFileName::Real(ref path) => {
            let mut file = File::create_buffered(path).unwrap();
            let _ = write!(file, "{}", krate);
        }
    }
}
