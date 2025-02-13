use std::fs::File;
use std::io::Write;

use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir_pretty::{self, AnnNode, Nested, PpAnn, State};
use rustc_middle::ty::print::with_crate_prefix;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::sym;

struct InterfaceAnn<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> PpAnn for InterfaceAnn<'tcx> {
    fn pre(&self, _state: &mut State<'_>, node: AnnNode<'_>) -> bool {
        match node {
            // FIXME: We can restrict more items here.
            AnnNode::Item(item) => {
                let def_id = item.owner_id.def_id;
                if self.tcx.has_attr(def_id, sym::prelude_import) {
                    return false;
                }
                match item.kind {
                    hir::ItemKind::ExternCrate(_) => {
                        if item.span.is_dummy() {
                            return false;
                        }
                        return true;
                    }

                    hir::ItemKind::Fn { .. } | hir::ItemKind::Impl(..) => {
                        return self.tcx.is_exportable(def_id.to_def_id());
                    }
                    _ => {}
                }
            }
            _ => {}
        };

        true
    }

    fn nested(&self, state: &mut State<'_>, nested: Nested) {
        self.tcx.nested(state, nested);
    }

    fn post(&self, state: &mut State<'_>, node: AnnNode<'_>) {
        if matches!(node, AnnNode::Item(..)) {
            state.word("\n");
        }
    }
}

fn vis_to_string(tcx: TyCtxt<'_>, vis: ty::Visibility, def_id: LocalDefId) -> String {
    match vis {
        ty::Visibility::Restricted(restricted_id) => {
            if restricted_id.is_top_level_module() {
                "pub(crate)".to_string()
            } else if restricted_id == tcx.parent_module_from_def_id(def_id).to_local_def_id() {
                "pub(self)".to_string()
            } else {
                let path = with_crate_prefix!(tcx.def_path_str(restricted_id));
                format!("pub(in {})", path)
            }
        }
        ty::Visibility::Public => "pub".to_string(),
    }
}

pub fn write_interface<'tcx>(tcx: TyCtxt<'tcx>) {
    if !(tcx.crate_types().contains(&rustc_session::config::CrateType::Sdylib)
        && tcx.sess.opts.output_types.should_codegen())
    {
        return;
    }
    let sess = tcx.sess;
    let _timer = sess.timer("write_interface");

    let ann = InterfaceAnn { tcx };
    let hir_map = tcx.hir();
    let attrs = |id| hir_map.attrs(id);

    let visibilities = |id| vis_to_string(tcx, tcx.visibility(id).expect_local(), id);

    let krate = rustc_hir_pretty::print_crate_as_interface(
        hir_map.root_module(),
        &attrs,
        &visibilities,
        &ann,
    );

    let export_output = tcx.output_filenames(()).interface_path();
    let mut file = File::create_buffered(export_output).unwrap();
    let _ = write!(file, "{}", krate);
}
