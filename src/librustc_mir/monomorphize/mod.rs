use rustc::traits;
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::ty::{self, Ty, TyCtxt, Instance};

pub use self::item::{MonoItem, MonoItemExt};

pub mod collector;
pub mod item;
pub mod partitioning;

#[inline(never)] // give this a place in the profiler
pub fn assert_symbols_are_distinct<'a, 'tcx, I>(tcx: TyCtxt<'a, 'tcx, 'tcx>, mono_items: I)
    where I: Iterator<Item=&'a MonoItem<'tcx>>
{
    let mut symbols: Vec<_> = mono_items.map(|mono_item| {
        (mono_item, mono_item.symbol_name(tcx))
    }).collect();

    symbols.sort_by_key(|sym| sym.1);

    for pair in symbols.windows(2) {
        let sym1 = &pair[0].1;
        let sym2 = &pair[1].1;

        if sym1 == sym2 {
            let mono_item1 = pair[0].0;
            let mono_item2 = pair[1].0;

            let span1 = mono_item1.local_span(tcx);
            let span2 = mono_item2.local_span(tcx);

            // Deterministically select one of the spans for error reporting
            let span = match (span1, span2) {
                (Some(span1), Some(span2)) => {
                    Some(if span1.lo().0 > span2.lo().0 {
                        span1
                    } else {
                        span2
                    })
                }
                (span1, span2) => span1.or(span2),
            };

            let error_message = format!("symbol `{}` is already defined", sym1);

            if let Some(span) = span {
                tcx.sess.span_fatal(span, &error_message)
            } else {
                tcx.sess.fatal(&error_message)
            }
        }
    }
}

pub fn custom_coerce_unsize_info<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           source_ty: Ty<'tcx>,
                                           target_ty: Ty<'tcx>)
                                           -> CustomCoerceUnsized {
    let def_id = tcx.lang_items().coerce_unsized_trait().unwrap();

    let trait_ref = ty::Binder::bind(ty::TraitRef {
        def_id: def_id,
        substs: tcx.mk_substs_trait(source_ty, &[target_ty.into()])
    });

    match tcx.codegen_fulfill_obligation( (ty::ParamEnv::reveal_all(), trait_ref)) {
        traits::VtableImpl(traits::VtableImplData { impl_def_id, .. }) => {
            tcx.coerce_unsized_info(impl_def_id).custom_kind.unwrap()
        }
        vtable => {
            bug!("invalid CoerceUnsized vtable: {:?}", vtable);
        }
    }
}
