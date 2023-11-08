use std::io;

use rustc_middle::ty::TyCtxt;
use stable_mir::{
    ty::{RigidTy, TyKind},
    CrateItem, mir::Mutability,
};


use super::{run, RustcInternal};

pub fn write_smir_pretty<'tcx>(tcx: TyCtxt<'tcx>, w: &mut dyn io::Write) -> io::Result<()> { 
    run(tcx, || {
        let items = stable_mir::all_local_items();
        items.iter().for_each(|item| {
            // Because we can't return a Result from a closure, we have to unwrap here.
            writeln!(w, "{}", function_name(*item,tcx)).unwrap();
            writeln!(w, "{}", function_body(*item,tcx)).unwrap();
        })
    });
    Ok(())
}

pub fn function_name(item: CrateItem,tcx: TyCtxt<'_>) -> String {
    let mut name = String::new();
    let body  = item.body();
    name.push_str("fn ");
    name.push_str(item.name().as_str());
    if body.arg_locals().is_empty() {
        name.push_str("()");
    }else{
        name.push_str("(");
    }
    body.arg_locals().iter().for_each(|local| {
        name.push_str(format!("_{}: ",local.local).as_str());
        name.push_str(&pretty_ty(local.ty.kind(), tcx));
    });
    if !body.arg_locals().is_empty() {
        name.push_str(")");
    }
    let return_local = body.ret_local();
    name.push_str(" -> ");
    name.push_str(&pretty_ty(return_local.ty.kind(), tcx));
    name.push_str(" {");
    name
}

pub fn function_body(item: CrateItem,_tcx: TyCtxt<'_>) -> String {
    let mut body_str = String::new();
    let body  = item.body();
    body.inner_locals().iter().for_each(|local| {
        body_str.push_str("    ");
        body_str.push_str(format!("let {}",ret_mutability(&local.mutability)).as_str());
        body_str.push_str(format!("_{}: ",local.local).as_str());
        body_str.push_str(format!("{}",pretty_ty(local.ty.kind(), _tcx)).as_str());
        body_str.push_str(";\n");

    });
    body_str.push_str("}");
    body_str

}

pub fn ret_mutability(mutability: &Mutability) -> String {
    match mutability {
        Mutability::Not => "".to_string(),
        Mutability::Mut => "mut ".to_string(),
    }
}

pub fn pretty_ty<'tcx>(ty: TyKind,tcx: TyCtxt<'tcx>) -> String {
    let mut pretty = String::new();
    pretty.push_str("");
    match ty {
        TyKind::RigidTy(rigid_ty) => match rigid_ty {
            RigidTy::Bool => "bool".to_string(),
            RigidTy::Char => "char".to_string(),
            RigidTy::Int(i) => match i {
                stable_mir::ty::IntTy::Isize => "isize".to_string(),
                stable_mir::ty::IntTy::I8 => "i8".to_string(),
                stable_mir::ty::IntTy::I16 => "i16".to_string(),
                stable_mir::ty::IntTy::I32 => "i32".to_string(),
                stable_mir::ty::IntTy::I64 => "i64".to_string(),
                stable_mir::ty::IntTy::I128 => "i128".to_string(),
            },
            RigidTy::Uint(u) => match u {
                stable_mir::ty::UintTy::Usize => "usize".to_string(),
                stable_mir::ty::UintTy::U8 => "u8".to_string(),
                stable_mir::ty::UintTy::U16 => "u16".to_string(),
                stable_mir::ty::UintTy::U32 => "u32".to_string(),
                stable_mir::ty::UintTy::U64 => "u64".to_string(),
                stable_mir::ty::UintTy::U128 => "u128".to_string(),
            },
            RigidTy::Float(f) => match f {
                stable_mir::ty::FloatTy::F32 => "f32".to_string(),
                stable_mir::ty::FloatTy::F64 => "f64".to_string(),
            },
            RigidTy::Adt(def, _) => format!("{:#?}", tcx.type_of(def.0.internal_via_tls()).instantiate_identity()),
            RigidTy::Foreign(_) => format!("{:#?}", rigid_ty),
            RigidTy::Str => "str".to_string(),
            RigidTy::Array(_ty, len) => {
                format!("[{};{:#?}]", 1,len.internal_via_tls())},
            RigidTy::Slice(ty) => pretty_ty(ty.kind(),tcx),
            RigidTy::RawPtr(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::Ref(_, ty, _) => pretty_ty(ty.kind(),tcx),
            RigidTy::FnDef(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::FnPtr(_) => format!("{:#?}", rigid_ty),
            RigidTy::Closure(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::Coroutine(_, _, _) => format!("{:#?}", rigid_ty),
            RigidTy::Dynamic(_, _, _) => format!("{:#?}", rigid_ty),
            RigidTy::Never => "!".to_string(),
            RigidTy::Tuple(tuple) => {
                if tuple.is_empty(){
                    "()".to_string()
                }else {
                    let mut tuple_str = String::new();
                    tuple_str.push_str("(");
                    tuple.iter().enumerate().for_each(|(i,ty)| {
                        tuple_str.push_str(&pretty_ty(ty.kind(),tcx));
                        if i != tuple.len() - 1 {
                            tuple_str.push_str(", ");
                        }
                    });
                    tuple_str.push_str(")");
                    tuple_str
                }
            },
        },
        TyKind::Alias(_, _) => format!("{:#?}", ty),
        TyKind::Param(_) => format!("{:#?}", ty),
        TyKind::Bound(_, _) => format!("{:#?}", ty),
    }
}
