use crate::ty::{self, TyCtxt, TypeFoldable};

use rustc_data_structures::fx::FxHashSet;
use syntax::symbol::InternedString;

use std::fmt;
use std::ops::Deref;

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

pub struct PrintCx<'a, 'gcx, 'tcx, P> {
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub printer: P,
    pub(crate) is_debug: bool,
    pub(crate) is_verbose: bool,
    pub(crate) identify_regions: bool,
    pub(crate) used_region_names: Option<FxHashSet<InternedString>>,
    pub(crate) region_index: usize,
    pub(crate) binder_depth: usize,
}

// HACK(eddyb) this is solely for `self: &mut PrintCx<Self>`, e.g. to
// implement traits on the printer and call the methods on the context.
impl<P> Deref for PrintCx<'_, '_, '_, P> {
    type Target = P;
    fn deref(&self) -> &P {
        &self.printer
    }
}

impl<P> PrintCx<'a, 'gcx, 'tcx, P> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>, printer: P) -> Self {
        PrintCx {
            tcx,
            printer,
            is_debug: false,
            is_verbose: tcx.sess.verbose(),
            identify_regions: tcx.sess.opts.debugging_opts.identify_regions,
            used_region_names: None,
            region_index: 0,
            binder_depth: 0,
        }
    }

    pub(crate) fn with<R>(printer: P, f: impl FnOnce(PrintCx<'_, '_, '_, P>) -> R) -> R {
        ty::tls::with(|tcx| f(PrintCx::new(tcx, printer)))
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

pub trait Print<'tcx, P> {
    type Output;

    fn print(&self, cx: &mut PrintCx<'_, '_, 'tcx, P>) -> Self::Output;
    fn print_display(&self, cx: &mut PrintCx<'_, '_, 'tcx, P>) -> Self::Output {
        let old_debug = cx.is_debug;
        cx.is_debug = false;
        let result = self.print(cx);
        cx.is_debug = old_debug;
        result
    }
    fn print_debug(&self, cx: &mut PrintCx<'_, '_, 'tcx, P>) -> Self::Output {
        let old_debug = cx.is_debug;
        cx.is_debug = true;
        let result = self.print(cx);
        cx.is_debug = old_debug;
        result
    }
}

pub struct FmtPrinter<F: fmt::Write> {
    pub fmt: F,
}
