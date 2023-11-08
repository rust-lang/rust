use std::io;

use rustc_middle::ty::TyCtxt;
use stable_mir::{
    mir::{Mutability, Operand, Rvalue, StatementKind},
    ty::{RigidTy, TyKind},
    CrateItem,
};

use super::{run, RustcInternal};

pub fn write_smir_pretty<'tcx>(tcx: TyCtxt<'tcx>, w: &mut dyn io::Write) -> io::Result<()> {
    run(tcx, || {
        let items = stable_mir::all_local_items();
        items.iter().for_each(|item| {
            // Because we can't return a Result from a closure, we have to unwrap here.
            writeln!(w, "{}", function_name(*item, tcx)).unwrap();
            writeln!(w, "{}", function_body(*item, tcx)).unwrap();
            writeln!(w, "------------------").unwrap();
            item.body().blocks.iter().for_each(|block| {
                block.statements.iter().for_each(|statement| {
                    writeln!(w, "{}", pretty_statement(&statement.kind, tcx)).unwrap();
                });
            })
        })
    });
    Ok(())
}

pub fn function_name(item: CrateItem, tcx: TyCtxt<'_>) -> String {
    let mut name = String::new();
    let body = item.body();
    name.push_str("fn ");
    name.push_str(item.name().as_str());
    if body.arg_locals().is_empty() {
        name.push_str("()");
    } else {
        name.push_str("(");
    }
    body.arg_locals().iter().for_each(|local| {
        name.push_str(format!("_{}: ", local.local).as_str());
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

pub fn function_body(item: CrateItem, _tcx: TyCtxt<'_>) -> String {
    let mut body_str = String::new();
    let body = item.body();
    body.inner_locals().iter().for_each(|local| {
        body_str.push_str("    ");
        body_str.push_str(format!("let {}", ret_mutability(&local.mutability)).as_str());
        body_str.push_str(format!("_{}: ", local.local).as_str());
        body_str.push_str(format!("{}", pretty_ty(local.ty.kind(), _tcx)).as_str());
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

pub fn pretty_statement(statement: &StatementKind, tcx: TyCtxt<'_>) -> String {
    let mut pretty = String::new();
    match statement {
        StatementKind::Assign(place, rval) => {
            pretty.push_str(format!("_{} = ", place.local).as_str());
            pretty.push_str(&pretty_rvalue(rval, tcx))
        }
        StatementKind::FakeRead(_, _) => todo!(),
        StatementKind::SetDiscriminant { .. } => todo!(),
        StatementKind::Deinit(_) => todo!(),
        StatementKind::StorageLive(_) => todo!(),
        StatementKind::StorageDead(_) => todo!(),
        StatementKind::Retag(_, _) => todo!(),
        StatementKind::PlaceMention(_) => todo!(),
        StatementKind::AscribeUserType { .. } => todo!(),
        StatementKind::Coverage(_) => todo!(),
        StatementKind::Intrinsic(_) => todo!(),
        StatementKind::ConstEvalCounter => (),
        StatementKind::Nop => (),
    }
    pretty
}

pub fn pretty_operand(operand: &Operand, _tcx: TyCtxt<'_>) -> String {
    let mut pretty = String::new();
    match operand {
        Operand::Copy(copy) => {
            pretty.push_str("");
            pretty.push_str(format!("{}", copy.local).as_str());
        }
        Operand::Move(mv) => {
            pretty.push_str("move");
            pretty.push_str(format!("{}", mv.local).as_str());
        }
        Operand::Constant(cnst) => {
            pretty.push_str("const ");
            pretty.push_str(cnst.literal.internal_via_tls().to_string().as_str());
        }
    }
    pretty
}

pub fn pretty_rvalue(rval: &Rvalue, tcx: TyCtxt<'_>) -> String {
    let mut pretty = String::new();
    match rval {
        Rvalue::AddressOf(muta, addr) => {
            pretty.push_str("address_of");
            pretty.push_str(&ret_mutability(&muta));
            pretty.push_str(format!("{}", addr.local).as_str());
        }
        Rvalue::Aggregate(aggregatekind, operands) => {
            pretty.push_str(format!("{:#?}", aggregatekind).as_str());
            pretty.push_str("(");
            operands.iter().enumerate().for_each(|(i, op)| {
                pretty.push_str(&pretty_operand(op, tcx));
                if i != operands.len() - 1 {
                    pretty.push_str(", ");
                }
            });
            pretty.push_str(")");
        }
        Rvalue::BinaryOp(bin, op, op2) => {
            pretty.push_str(&pretty_operand(op, tcx));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", bin).as_str());
            pretty.push_str(" ");
            pretty.push_str(&pretty_operand(op2, tcx));
        }
        Rvalue::Cast(_, op, ty) => {
            pretty.push_str(&pretty_operand(op, tcx));
            pretty.push_str(" as ");
            pretty.push_str(&pretty_ty(ty.kind(), tcx));
        }
        Rvalue::CheckedBinaryOp(bin, op1, op2) => {
            pretty.push_str(&pretty_operand(op1, tcx));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", bin).as_str());
            pretty.push_str(" ");
            pretty.push_str(&pretty_operand(op2, tcx));
        }
        Rvalue::CopyForDeref(deref) => {
            pretty.push_str("CopyForDeref");
            pretty.push_str(format!("{}", deref.local).as_str());
        }
        Rvalue::Discriminant(place) => {
            pretty.push_str("discriminant");
            pretty.push_str(format!("{}", place.local).as_str());
        }
        Rvalue::Len(len) => {
            pretty.push_str("len");
            pretty.push_str(format!("{}", len.local).as_str());
        }
        Rvalue::Ref(_, borrowkind, place) => {
            pretty.push_str("ref");
            pretty.push_str(format!("{:#?}", borrowkind).as_str());
            pretty.push_str(format!("{}", place.local).as_str());
        }
        Rvalue::Repeat(op, cnst) => {
            pretty.push_str(&pretty_operand(op, tcx));
            pretty.push_str(" ");
            pretty.push_str(&pretty_ty(cnst.ty().kind(), tcx));
        }
        Rvalue::ShallowInitBox(_, _) => todo!(),
        Rvalue::ThreadLocalRef(item) => {
            pretty.push_str("thread_local_ref");
            pretty.push_str(format!("{:#?}", item).as_str());
        }
        Rvalue::NullaryOp(nul, ty) => {
            pretty.push_str(format!("{:#?}", nul).as_str());
            pretty.push_str(&&pretty_ty(ty.kind(), tcx));
            pretty.push_str(" ");
        }
        Rvalue::UnaryOp(un, op) => {
            pretty.push_str(&pretty_operand(op, tcx));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", un).as_str());
        }
        Rvalue::Use(op) => pretty.push_str(&pretty_operand(op, tcx)),
    }
    pretty
}

pub fn pretty_ty(ty: TyKind, tcx: TyCtxt<'_>) -> String {
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
            RigidTy::Adt(def, _) => {
                format!("{:#?}", tcx.type_of(def.0.internal_via_tls()).instantiate_identity())
            }
            RigidTy::Foreign(_) => format!("{:#?}", rigid_ty),
            RigidTy::Str => "str".to_string(),
            RigidTy::Array(_ty, len) => {
                format!("[{};{:#?}]", 1, len.internal_via_tls())
            }
            RigidTy::Slice(ty) => pretty_ty(ty.kind(), tcx),
            RigidTy::RawPtr(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::Ref(_, ty, _) => pretty_ty(ty.kind(), tcx),
            RigidTy::FnDef(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::FnPtr(_) => format!("{:#?}", rigid_ty),
            RigidTy::Closure(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::Coroutine(_, _, _) => format!("{:#?}", rigid_ty),
            RigidTy::Dynamic(_, _, _) => format!("{:#?}", rigid_ty),
            RigidTy::Never => "!".to_string(),
            RigidTy::Tuple(tuple) => {
                if tuple.is_empty() {
                    "()".to_string()
                } else {
                    let mut tuple_str = String::new();
                    tuple_str.push_str("(");
                    tuple.iter().enumerate().for_each(|(i, ty)| {
                        tuple_str.push_str(&pretty_ty(ty.kind(), tcx));
                        if i != tuple.len() - 1 {
                            tuple_str.push_str(", ");
                        }
                    });
                    tuple_str.push_str(")");
                    tuple_str
                }
            }
        },
        TyKind::Alias(_, _) => format!("{:#?}", ty),
        TyKind::Param(_) => format!("{:#?}", ty),
        TyKind::Bound(_, _) => format!("{:#?}", ty),
    }
}
