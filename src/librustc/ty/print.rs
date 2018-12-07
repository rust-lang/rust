use crate::ty::{self, TyCtxt, TypeFoldable};

use rustc_data_structures::fx::FxHashSet;
use syntax::symbol::InternedString;

use std::fmt;

// FIXME(eddyb) this module uses `pub(crate)` for things used only
// from `ppaux` - when that is removed, they can be re-privatized.

struct LateBoundRegionNameCollector(FxHashSet<InternedString>);
impl<'tcx> ty::fold::TypeVisitor<'tcx> for LateBoundRegionNameCollector {
    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        match *r {
            ty::ReLateBound(_, ty::BrNamed(_, name)) => {
                self.0.insert(name);
            },
            _ => {},
        }
        r.super_visit_with(self)
    }
}

pub struct PrintCx<'a, 'gcx, 'tcx> {
    pub(crate) tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub(crate) is_debug: bool,
    pub(crate) is_verbose: bool,
    pub(crate) identify_regions: bool,
    pub(crate) used_region_names: Option<FxHashSet<InternedString>>,
    pub(crate) region_index: usize,
    pub(crate) binder_depth: usize,
}

impl PrintCx<'a, 'gcx, 'tcx> {
    pub(crate) fn with<R>(f: impl FnOnce(PrintCx<'_, '_, '_>) -> R) -> R {
        ty::tls::with(|tcx| {
            f(PrintCx {
                tcx,
                is_debug: false,
                is_verbose: tcx.sess.verbose(),
                identify_regions: tcx.sess.opts.debugging_opts.identify_regions,
                used_region_names: None,
                region_index: 0,
                binder_depth: 0,
            })
        })
    }
    pub(crate) fn prepare_late_bound_region_info<T>(&mut self, value: &ty::Binder<T>)
    where T: TypeFoldable<'tcx>
    {
        let mut collector = LateBoundRegionNameCollector(Default::default());
        value.visit_with(&mut collector);
        self.used_region_names = Some(collector.0);
        self.region_index = 0;
    }
}

pub trait Print<'tcx> {
    fn print<F: fmt::Write>(&self, f: &mut F, cx: &mut PrintCx<'_, '_, 'tcx>) -> fmt::Result;
    fn print_to_string(&self, cx: &mut PrintCx<'_, '_, 'tcx>) -> String {
        let mut result = String::new();
        let _ = self.print(&mut result, cx);
        result
    }
    fn print_display<F: fmt::Write>(
        &self,
        f: &mut F,
        cx: &mut PrintCx<'_, '_, 'tcx>,
    ) -> fmt::Result {
        let old_debug = cx.is_debug;
        cx.is_debug = false;
        let result = self.print(f, cx);
        cx.is_debug = old_debug;
        result
    }
    fn print_display_to_string(&self, cx: &mut PrintCx<'_, '_, 'tcx>) -> String {
        let mut result = String::new();
        let _ = self.print_display(&mut result, cx);
        result
    }
    fn print_debug<F: fmt::Write>(&self, f: &mut F, cx: &mut PrintCx<'_, '_, 'tcx>) -> fmt::Result {
        let old_debug = cx.is_debug;
        cx.is_debug = true;
        let result = self.print(f, cx);
        cx.is_debug = old_debug;
        result
    }
    fn print_debug_to_string(&self, cx: &mut PrintCx<'_, '_, 'tcx>) -> String {
        let mut result = String::new();
        let _ = self.print_debug(&mut result, cx);
        result
    }
}
