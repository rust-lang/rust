//! Detecting usage of the `#[debugger_visualizer]` attribute.

use rustc_ast::Attribute;
use rustc_expand::base::resolve_path;
use rustc_middle::middle::debugger_visualizer::{DebuggerVisualizerFile, DebuggerVisualizerType};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::sym;

use crate::errors::{DebugVisualizerInvalid, DebugVisualizerUnreadable};

impl DebuggerVisualizerCollector<'_> {
    fn check_for_debugger_visualizer(&mut self, attr: &Attribute) {
        if attr.has_name(sym::debugger_visualizer) {
            let Some(hints) = attr.meta_item_list() else {
                self.sess.dcx().emit_err(DebugVisualizerInvalid { span: attr.span });
                return;
            };

            let [hint] = hints.as_slice() else {
                self.sess.dcx().emit_err(DebugVisualizerInvalid { span: attr.span });
                return;
            };

            let Some(meta_item) = hint.meta_item() else {
                self.sess.dcx().emit_err(DebugVisualizerInvalid { span: attr.span });
                return;
            };

            let (visualizer_type, visualizer_path) = match (meta_item.name(), meta_item.value_str())
            {
                (Some(sym::natvis_file), Some(value)) => (DebuggerVisualizerType::Natvis, value),
                (Some(sym::gdb_script_file), Some(value)) => {
                    (DebuggerVisualizerType::GdbPrettyPrinter, value)
                }
                (_, _) => {
                    self.sess.dcx().emit_err(DebugVisualizerInvalid { span: meta_item.span });
                    return;
                }
            };

            let file = match resolve_path(&self.sess, visualizer_path.as_str(), attr.span) {
                Ok(file) => file,
                Err(err) => {
                    err.emit();
                    return;
                }
            };

            match self.sess.source_map().load_binary_file(&file) {
                Ok((source, _)) => {
                    self.visualizers.push(DebuggerVisualizerFile::new(
                        source,
                        visualizer_type,
                        file,
                    ));
                }
                Err(error) => {
                    self.sess.dcx().emit_err(DebugVisualizerUnreadable {
                        span: meta_item.span,
                        file: &file,
                        error,
                    });
                }
            }
        }
    }
}

struct DebuggerVisualizerCollector<'a> {
    sess: &'a Session,
    visualizers: Vec<DebuggerVisualizerFile>,
}

impl<'ast> rustc_ast::visit::Visitor<'ast> for DebuggerVisualizerCollector<'_> {
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.check_for_debugger_visualizer(attr);
        rustc_ast::visit::walk_attribute(self, attr);
    }
}

/// Traverses and collects the debugger visualizers for a specific crate.
fn debugger_visualizers(tcx: TyCtxt<'_>, _: LocalCrate) -> Vec<DebuggerVisualizerFile> {
    let resolver_and_krate = tcx.resolver_for_lowering().borrow();
    let krate = &*resolver_and_krate.1;

    let mut visitor = DebuggerVisualizerCollector { sess: tcx.sess, visualizers: Vec::new() };
    rustc_ast::visit::Visitor::visit_crate(&mut visitor, krate);

    // We are collecting visualizers in AST-order, which is deterministic,
    // so we don't need to do any explicit sorting in order to get a
    // deterministic query result
    visitor.visualizers
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.debugger_visualizers = debugger_visualizers;
}
