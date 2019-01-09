use core::DocContext;

use super::*;

pub fn get_def_from_def_id<F>(cx: &DocContext,
                              def_id: DefId,
                              callback: &F,
) -> Vec<Item>
where F: Fn(& dyn Fn(DefId) -> Def) -> Vec<Item> {
    let ty = cx.tcx.type_of(def_id);

    match ty.sty {
        ty::Adt(adt, _) => callback(&match adt.adt_kind() {
            AdtKind::Struct => Def::Struct,
            AdtKind::Enum => Def::Enum,
            AdtKind::Union => Def::Union,
        }),
        ty::Int(_) |
        ty::Uint(_) |
        ty::Float(_) |
        ty::Str |
        ty::Bool |
        ty::Char => callback(&move |_: DefId| {
            match ty.sty {
                ty::Int(x) => Def::PrimTy(hir::Int(x)),
                ty::Uint(x) => Def::PrimTy(hir::Uint(x)),
                ty::Float(x) => Def::PrimTy(hir::Float(x)),
                ty::Str => Def::PrimTy(hir::Str),
                ty::Bool => Def::PrimTy(hir::Bool),
                ty::Char => Def::PrimTy(hir::Char),
                _ => unreachable!(),
            }
        }),
        _ => {
            debug!("Unexpected type {:?}", def_id);
            Vec::new()
        }
    }
}

pub fn get_def_from_node_id<F>(cx: &DocContext,
                               id: ast::NodeId,
                               name: String,
                               callback: &F,
) -> Vec<Item>
where F: Fn(& dyn Fn(DefId) -> Def, String) -> Vec<Item> {
    let item = &cx.tcx.hir().expect_item(id).node;

    callback(&match *item {
        hir::ItemKind::Struct(_, _) => Def::Struct,
        hir::ItemKind::Union(_, _) => Def::Union,
        hir::ItemKind::Enum(_, _) => Def::Enum,
        _ => panic!("Unexpected type {:?} {:?}", item, id),
    }, name)
}
