use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub fn provide(providers: &mut Providers) {
    *providers = Providers { layout_generators, ..*providers };
}

/// Computes the generators present in the layout of a type.
/// This expects a normalized input type with regions erased.
fn layout_generators<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> &'tcx ty::List<Ty<'tcx>> {
    let mut generators = Vec::new();

    let inner = |generators: &mut Vec<_>, ty: Ty<'tcx>| {
        let list = tcx.layout_generators(ty);
        for generator in list.iter() {
            generators.push(generator);
        }
    };

    match *ty.kind() {
        // These can't contain generators in their layout
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::FnPtr(_)
        | ty::FnDef(..)
        | ty::Never
        | ty::Ref(..)
        | ty::RawPtr(..)
        | ty::Str => {}

        ty::Array(element, _) => {
            inner(&mut generators, element);
        }

        ty::Generator(..) => {
            generators.push(ty);
        }

        ty::Closure(_, ref args) => {
            let tys = args.as_closure().upvar_tys();
            tys.iter().for_each(|ty| inner(&mut generators, ty));
        }

        ty::Tuple(tys) => {
            tys.iter().for_each(|ty| inner(&mut generators, ty));
        }

        ty::Adt(def, args) => {
            def.variants().iter().for_each(|v| {
                v.fields.iter().for_each(|field| {
                    let ty = field.ty(tcx, args);
                    let ty = tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), ty);
                    inner(&mut generators, ty)
                })
            });
        }

        ty::Slice(..) | ty::Dynamic(..) | ty::Foreign(..) => {
            bug!("these are unsized")
        }

        ty::Alias(..)
        | ty::Bound(..)
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::Infer(_)
        | ty::Error(_)
        | ty::Placeholder(..)
        | ty::Param(_) => {
            bug!("unexpected type")
        }
    }

    tcx.mk_type_list(&generators)
}
