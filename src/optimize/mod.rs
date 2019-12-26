use crate::prelude::*;

mod stack2reg;

pub fn optimize_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    func: &mut Function,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
) {
    self::stack2reg::optimize_function(func, clif_comments, format!("{:?}", instance));
    #[cfg(debug_assertions)]
    crate::pretty_clif::write_clif_file(tcx, "stack2reg", instance, &*func, &*clif_comments, None);
    crate::base::verify_func(tcx, &*clif_comments, &*func);
}
