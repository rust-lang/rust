use rustc_span::def_id::DefIndex;

use super::*;

macro_rules! define_tests {
    ($($name:ident $kind:ident $variant:ident {$($init:tt)*})*) => {$(
        #[test]
        fn $name() {
            let unambig = $kind::$variant::<'_, ()> { $($init)* };
            let unambig_to_ambig = unsafe { std::mem::transmute::<_, $kind<'_, AmbigArg>>(unambig) };

            assert!(matches!(&unambig_to_ambig, &$kind::$variant { $($init)* }));

            let ambig_to_unambig = unsafe { std::mem::transmute::<_, $kind<'_, ()>>(unambig_to_ambig) };

            assert!(matches!(&ambig_to_unambig, &$kind::$variant { $($init)* }));
        }
    )*};
}

define_tests! {
    cast_never TyKind Never {}
    cast_tup TyKind Tup { 0: &[Ty { span: DUMMY_SP, hir_id: HirId::INVALID, kind: TyKind::Never }] }
    cast_ptr TyKind Ptr { 0: MutTy { ty: &Ty { span: DUMMY_SP, hir_id: HirId::INVALID, kind: TyKind::Never }, mutbl: Mutability::Not }}
    cast_array TyKind Array {
        0: &Ty { span: DUMMY_SP, hir_id: HirId::INVALID, kind: TyKind::Never },
        1: &ConstArg { hir_id: HirId::INVALID, kind: ConstArgKind::Anon(&AnonConst {
            hir_id: HirId::INVALID,
            def_id: LocalDefId { local_def_index: DefIndex::ZERO },
            body: BodyId { hir_id: HirId::INVALID },
            span: DUMMY_SP,
        })}
    }

    cast_anon ConstArgKind Anon {
        0: &AnonConst {
            hir_id: HirId::INVALID,
            def_id: LocalDefId { local_def_index: DefIndex::ZERO },
            body: BodyId { hir_id: HirId::INVALID },
            span: DUMMY_SP,
        }
    }
}

#[test]
fn trait_object_roundtrips() {
    trait_object_roundtrips_impl(TraitObjectSyntax::Dyn);
    trait_object_roundtrips_impl(TraitObjectSyntax::None);
}

fn trait_object_roundtrips_impl(syntax: TraitObjectSyntax) {
    let lt = Lifetime {
        hir_id: HirId::INVALID,
        ident: Ident::new(sym::name, DUMMY_SP),
        kind: LifetimeKind::Static,
        source: LifetimeSource::Other,
        syntax: LifetimeSyntax::Implicit,
    };
    let unambig = TyKind::TraitObject::<'_, ()>(&[], TaggedRef::new(&lt, syntax));
    let unambig_to_ambig = unsafe { std::mem::transmute::<_, TyKind<'_, AmbigArg>>(unambig) };

    match unambig_to_ambig {
        TyKind::TraitObject(_, tagged_ref) => {
            assert!(tagged_ref.tag() == syntax)
        }
        _ => panic!("`TyKind::TraitObject` did not roundtrip"),
    };

    let ambig_to_unambig = unsafe { std::mem::transmute::<_, TyKind<'_, ()>>(unambig_to_ambig) };

    match ambig_to_unambig {
        TyKind::TraitObject(_, tagged_ref) => {
            assert!(tagged_ref.tag() == syntax)
        }
        _ => panic!("`TyKind::TraitObject` did not roundtrip"),
    };
}
