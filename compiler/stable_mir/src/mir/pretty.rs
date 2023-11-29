use crate::crate_def::CrateDef;
use crate::mir::{Operand, Rvalue, StatementKind, UnwindAction};
use crate::ty::{DynKind, FloatTy, IntTy, RigidTy, TyKind, UintTy};
use crate::{with, Body, CrateItem, Mutability};
use std::io::Write;
use std::{io, iter};

use super::{AssertMessage, BinOp, TerminatorKind};

pub fn function_name(item: CrateItem) -> String {
    let mut pretty_name = String::new();
    let body = item.body();
    pretty_name.push_str("fn ");
    pretty_name.push_str(item.name().as_str());
    if body.arg_locals().is_empty() {
        pretty_name.push_str("()");
    } else {
        pretty_name.push_str("(");
    }
    body.arg_locals().iter().enumerate().for_each(|(index, local)| {
        pretty_name.push_str(format!("_{}: ", index).as_str());
        pretty_name.push_str(&pretty_ty(local.ty.kind()));
    });
    if !body.arg_locals().is_empty() {
        pretty_name.push_str(")");
    }
    let return_local = body.ret_local();
    pretty_name.push_str(" -> ");
    pretty_name.push_str(&pretty_ty(return_local.ty.kind()));
    pretty_name.push_str(" {");
    pretty_name
}

pub fn function_body(body: &Body) -> String {
    let mut pretty_body = String::new();
    body.inner_locals().iter().enumerate().for_each(|(index, local)| {
        pretty_body.push_str("    ");
        pretty_body.push_str(format!("let {}", ret_mutability(&local.mutability)).as_str());
        pretty_body.push_str(format!("_{}: ", index).as_str());
        pretty_body.push_str(format!("{}", pretty_ty(local.ty.kind())).as_str());
        pretty_body.push_str(";\n");
    });
    pretty_body.push_str("}");
    pretty_body
}

pub fn ret_mutability(mutability: &Mutability) -> String {
    match mutability {
        Mutability::Not => "".to_string(),
        Mutability::Mut => "mut ".to_string(),
    }
}

pub fn pretty_statement(statement: &StatementKind) -> String {
    let mut pretty = String::new();
    match statement {
        StatementKind::Assign(place, rval) => {
            pretty.push_str(format!("        _{} = ", place.local).as_str());
            pretty.push_str(format!("{}", &pretty_rvalue(rval)).as_str());
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

pub fn pretty_terminator<W: io::Write>(terminator: &TerminatorKind, w: &mut W) -> io::Result<()> {
    write!(w, "{}", pretty_terminator_head(terminator))?;
    let successors = terminator.successors();
    let successor_count = successors.len();
    let labels = pretty_successor_labels(terminator);

    let show_unwind = !matches!(terminator.unwind(), None | Some(UnwindAction::Cleanup(_)));
    let fmt_unwind = |fmt: &mut dyn Write| -> io::Result<()> {
        write!(fmt, "unwind ")?;
        match terminator.unwind() {
            None | Some(UnwindAction::Cleanup(_)) => unreachable!(),
            Some(UnwindAction::Continue) => write!(fmt, "continue"),
            Some(UnwindAction::Unreachable) => write!(fmt, "unreachable"),
            Some(UnwindAction::Terminate) => write!(fmt, "terminate"),
        }
    };

    match (successor_count, show_unwind) {
        (0, false) => Ok(()),
        (0, true) => {
            write!(w, " -> ")?;
            fmt_unwind(w)?;
            Ok(())
        }
        (1, false) => {
            write!(w, " -> {:?}", successors[0])?;
            Ok(())
        }
        _ => {
            write!(w, " -> [")?;
            for (i, target) in successors.iter().enumerate() {
                if i > 0 {
                    write!(w, ", ")?;
                }
                write!(w, "{}: bb{:?}", labels[i], target)?;
            }
            if show_unwind {
                write!(w, ", ")?;
                fmt_unwind(w)?;
            }
            write!(w, "]")
        }
    }?;

    Ok(())
}

pub fn pretty_terminator_head(terminator: &TerminatorKind) -> String {
    use self::TerminatorKind::*;
    let mut pretty = String::new();
    match terminator {
        Goto { .. } => format!("        goto"),
        SwitchInt { discr, .. } => {
            format!("        switchInt(_{})", pretty_operand(discr))
        }
        Resume => format!("        resume"),
        Abort => format!("        abort"),
        Return => format!("        return"),
        Unreachable => format!("        unreachable"),
        Drop { place, .. } => format!("        drop(_{:?})", place.local),
        Call { func, args, destination, .. } => {
            pretty.push_str("        ");
            pretty.push_str(format!("_{} = ", destination.local).as_str());
            pretty.push_str(&pretty_operand(func));
            pretty.push_str("(");
            args.iter().enumerate().for_each(|(i, arg)| {
                if i > 0 {
                    pretty.push_str(", ");
                }
                pretty.push_str(&pretty_operand(arg));
            });
            pretty.push_str(")");
            pretty
        }
        Assert { cond, expected, msg, target: _, unwind: _ } => {
            pretty.push_str("        assert(");
            if !expected {
                pretty.push_str("!");
            }
            pretty.push_str(format!("{} bool),", &pretty_operand(cond)).as_str());
            pretty.push_str(&pretty_assert_message(msg));
            pretty.push_str(")");
            pretty
        }
        InlineAsm { .. } => todo!(),
    }
}

pub fn pretty_successor_labels(terminator: &TerminatorKind) -> Vec<String> {
    use self::TerminatorKind::*;
    match terminator {
        Resume | Abort | Return | Unreachable => vec![],
        Goto { .. } => vec!["".to_string()],
        SwitchInt { targets, .. } => targets
            .branches()
            .map(|(val, _target)| format!("{val}"))
            .chain(iter::once("otherwise".into()))
            .collect(),
        Drop { unwind: UnwindAction::Cleanup(_), .. } => vec!["return".into(), "unwind".into()],
        Drop { unwind: _, .. } => vec!["return".into()],
        Call { target: Some(_), unwind: UnwindAction::Cleanup(_), .. } => {
            vec!["return".into(), "unwind".into()]
        }
        Call { target: Some(_), unwind: _, .. } => vec!["return".into()],
        Call { target: None, unwind: UnwindAction::Cleanup(_), .. } => vec!["unwind".into()],
        Call { target: None, unwind: _, .. } => vec![],
        Assert { unwind: UnwindAction::Cleanup(_), .. } => {
            vec!["success".into(), "unwind".into()]
        }
        Assert { unwind: _, .. } => vec!["success".into()],
        InlineAsm { .. } => todo!(),
    }
}

pub fn pretty_assert_message(msg: &AssertMessage) -> String {
    let mut pretty = String::new();
    match msg {
        AssertMessage::BoundsCheck { len, index } => {
            let pretty_len = pretty_operand(len);
            let pretty_index = pretty_operand(index);
            pretty.push_str(format!("\"index out of bounds: the length is {{}} but the index is {{}}\", {pretty_len}, {pretty_index}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Add, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            pretty.push_str(format!("\"attempt to compute `{{}} + {{}}`, which would overflow\", {pretty_l}, {pretty_r}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Sub, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            pretty.push_str(format!("\"attempt to compute `{{}} - {{}}`, which would overflow\", {pretty_l}, {pretty_r}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Mul, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            pretty.push_str(format!("\"attempt to compute `{{}} * {{}}`, which would overflow\", {pretty_l}, {pretty_r}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Div, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            pretty.push_str(format!("\"attempt to compute `{{}} / {{}}`, which would overflow\", {pretty_l}, {pretty_r}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Rem, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            pretty.push_str(format!("\"attempt to compute `{{}} % {{}}`, which would overflow\", {pretty_l}, {pretty_r}").as_str());
            pretty
        }
        AssertMessage::Overflow(BinOp::Shr, _, r) => {
            let pretty_r = pretty_operand(r);
            pretty.push_str(
                format!("\"attempt to shift right by `{{}}`, which would overflow\", {pretty_r}")
                    .as_str(),
            );
            pretty
        }
        AssertMessage::Overflow(BinOp::Shl, _, r) => {
            let pretty_r = pretty_operand(r);
            pretty.push_str(
                format!("\"attempt to shift left by `{{}}`, which would overflow\", {pretty_r}")
                    .as_str(),
            );
            pretty
        }
        AssertMessage::OverflowNeg(op) => {
            let pretty_op = pretty_operand(op);
            pretty.push_str(
                format!("\"attempt to negate `{{}}`, which would overflow\", {pretty_op}").as_str(),
            );
            pretty
        }
        AssertMessage::DivisionByZero(op) => {
            let pretty_op = pretty_operand(op);
            pretty.push_str(format!("\"attempt to divide `{{}}` by zero\", {pretty_op}").as_str());
            pretty
        }
        AssertMessage::RemainderByZero(op) => {
            let pretty_op = pretty_operand(op);
            pretty.push_str(
                format!("\"attempt to calculate the remainder of `{{}}` with a divisor of zero\", {pretty_op}").as_str(),
            );
            pretty
        }
        AssertMessage::ResumedAfterReturn(_) => {
            format!("attempt to resume a generator after completion")
        }
        AssertMessage::ResumedAfterPanic(_) => format!("attempt to resume a panicked generator"),
        AssertMessage::MisalignedPointerDereference { required, found } => {
            let pretty_required = pretty_operand(required);
            let pretty_found = pretty_operand(found);
            pretty.push_str(format!("\"misaligned pointer dereference: address must be a multiple of {{}} but is {{}}\",{pretty_required}, {pretty_found}").as_str());
            pretty
        }
        _ => todo!(),
    }
}

pub fn pretty_operand(operand: &Operand) -> String {
    let mut pretty = String::new();
    match operand {
        Operand::Copy(copy) => {
            pretty.push_str("");
            pretty.push_str(format!("{}", copy.local).as_str());
        }
        Operand::Move(mv) => {
            pretty.push_str("move ");
            pretty.push_str(format!("_{}", mv.local).as_str());
        }
        Operand::Constant(cnst) => {
            pretty.push_str("const ");
            pretty.push_str(with(|cx| cx.const_literal(&cnst.literal)).as_str());
        }
    }
    pretty
}

pub fn pretty_rvalue(rval: &Rvalue) -> String {
    let mut pretty = String::new();
    match rval {
        Rvalue::AddressOf(muta, addr) => {
            pretty.push_str("&raw ");
            pretty.push_str(&ret_mutability(muta));
            pretty.push_str(format!("(*_{})", addr.local).as_str());
        }
        Rvalue::Aggregate(aggregatekind, operands) => {
            pretty.push_str(format!("{:#?}", aggregatekind).as_str());
            pretty.push_str("(");
            operands.iter().enumerate().for_each(|(i, op)| {
                pretty.push_str(&pretty_operand(op));
                if i != operands.len() - 1 {
                    pretty.push_str(", ");
                }
            });
            pretty.push_str(")");
        }
        Rvalue::BinaryOp(bin, op, op2) => {
            pretty.push_str(&pretty_operand(op));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", bin).as_str());
            pretty.push_str(" ");
            pretty.push_str(&pretty_operand(op2));
        }
        Rvalue::Cast(_, op, ty) => {
            pretty.push_str(&pretty_operand(op));
            pretty.push_str(" as ");
            pretty.push_str(&pretty_ty(ty.kind()));
        }
        Rvalue::CheckedBinaryOp(bin, op1, op2) => {
            pretty.push_str(&pretty_operand(op1));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", bin).as_str());
            pretty.push_str(" ");
            pretty.push_str(&pretty_operand(op2));
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
            pretty.push_str(&pretty_operand(op));
            pretty.push_str(" ");
            pretty.push_str(&pretty_ty(cnst.ty().kind()));
        }
        Rvalue::ShallowInitBox(_, _) => todo!(),
        Rvalue::ThreadLocalRef(item) => {
            pretty.push_str("thread_local_ref");
            pretty.push_str(format!("{:#?}", item).as_str());
        }
        Rvalue::NullaryOp(nul, ty) => {
            pretty.push_str(format!("{:#?}", nul).as_str());
            pretty.push_str(&pretty_ty(ty.kind()));
            pretty.push_str(" ");
        }
        Rvalue::UnaryOp(un, op) => {
            pretty.push_str(&pretty_operand(op));
            pretty.push_str(" ");
            pretty.push_str(format!("{:#?}", un).as_str());
        }
        Rvalue::Use(op) => pretty.push_str(&pretty_operand(op)),
    }
    pretty
}

pub fn pretty_ty(ty: TyKind) -> String {
    let mut pretty = String::new();
    match ty {
        TyKind::RigidTy(rigid_ty) => match rigid_ty {
            RigidTy::Bool => "bool".to_string(),
            RigidTy::Char => "char".to_string(),
            RigidTy::Int(i) => match i {
                IntTy::Isize => "isize".to_string(),
                IntTy::I8 => "i8".to_string(),
                IntTy::I16 => "i16".to_string(),
                IntTy::I32 => "i32".to_string(),
                IntTy::I64 => "i64".to_string(),
                IntTy::I128 => "i128".to_string(),
            },
            RigidTy::Uint(u) => match u {
                UintTy::Usize => "usize".to_string(),
                UintTy::U8 => "u8".to_string(),
                UintTy::U16 => "u16".to_string(),
                UintTy::U32 => "u32".to_string(),
                UintTy::U64 => "u64".to_string(),
                UintTy::U128 => "u128".to_string(),
            },
            RigidTy::Float(f) => match f {
                FloatTy::F32 => "f32".to_string(),
                FloatTy::F64 => "f64".to_string(),
            },
            RigidTy::Adt(def, _) => {
                format!("{:#?}", with(|cx| cx.def_ty(def.0)))
            }
            RigidTy::Str => "str".to_string(),
            RigidTy::Array(ty, len) => {
                format!("[{}; {}]", pretty_ty(ty.kind()), with(|cx| cx.const_literal(&len)))
            }
            RigidTy::Slice(ty) => {
                format!("[{}]", pretty_ty(ty.kind()))
            }
            RigidTy::RawPtr(ty, mutability) => {
                pretty.push_str("*");
                match mutability {
                    Mutability::Not => pretty.push_str("const "),
                    Mutability::Mut => pretty.push_str("mut "),
                }
                pretty.push_str(&pretty_ty(ty.kind()));
                pretty
            }
            RigidTy::Ref(_, ty, mutability) => match mutability {
                Mutability::Not => format!("&{}", pretty_ty(ty.kind())),
                Mutability::Mut => format!("&mut {}", pretty_ty(ty.kind())),
            },
            RigidTy::FnDef(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::FnPtr(_) => format!("{:#?}", rigid_ty),
            RigidTy::Closure(_, _) => format!("{:#?}", rigid_ty),
            RigidTy::Coroutine(_, _, _) => format!("{:#?}", rigid_ty),
            RigidTy::Dynamic(data, region, repr) => {
                // FIXME: Fix binder printing, it looks ugly now
                pretty.push_str("(");
                match repr {
                    DynKind::Dyn => pretty.push_str("dyn "),
                    DynKind::DynStar => pretty.push_str("dyn* "),
                }
                pretty.push_str(format!("{:#?}", data).as_str());
                pretty.push_str(format!(" +  {:#?} )", region).as_str());
                pretty
            }
            RigidTy::Never => "!".to_string(),
            RigidTy::Tuple(tuple) => {
                if tuple.is_empty() {
                    "()".to_string()
                } else {
                    let mut tuple_str = String::new();
                    tuple_str.push_str("(");
                    tuple.iter().enumerate().for_each(|(i, ty)| {
                        tuple_str.push_str(&pretty_ty(ty.kind()));
                        if i != tuple.len() - 1 {
                            tuple_str.push_str(", ");
                        }
                    });
                    tuple_str.push_str(")");
                    tuple_str
                }
            }
            _ => format!("{:#?}", rigid_ty),
        },
        TyKind::Alias(_, _) => format!("{:#?}", ty),
        TyKind::Param(param_ty) => {
            format!("{:#?}", param_ty.name)
        }
        TyKind::Bound(_, _) => format!("{:#?}", ty),
    }
}
