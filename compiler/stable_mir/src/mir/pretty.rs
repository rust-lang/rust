use std::fmt::Debug;
use std::io::Write;
use std::{fmt, io, iter};

use fmt::{Display, Formatter};

use super::{AssertMessage, BinOp, BorrowKind, FakeBorrowKind, TerminatorKind};
use crate::mir::{Operand, Place, Rvalue, StatementKind, UnwindAction, VarDebugInfoContents};
use crate::ty::{IndexedVal, MirConst, Ty, TyConst};
use crate::{Body, Mutability, with};

impl Display for Ty {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        with(|ctx| write!(f, "{}", ctx.ty_pretty(*self)))
    }
}

impl Debug for Place {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        with(|ctx| write!(f, "{}", ctx.place_pretty(self)))
    }
}

pub(crate) fn function_body<W: Write>(writer: &mut W, body: &Body, name: &str) -> io::Result<()> {
    write!(writer, "fn {}(", name)?;
    body.arg_locals()
        .iter()
        .enumerate()
        .try_for_each(|(index, local)| write!(writer, "_{}: {}", index + 1, local.ty))?;
    write!(writer, ")")?;

    let return_local = body.ret_local();
    writeln!(writer, " -> {} {{", return_local.ty)?;

    body.locals().iter().enumerate().try_for_each(|(index, local)| -> io::Result<()> {
        if index == 0 || index > body.arg_count {
            writeln!(writer, "    let {}_{}: {};", pretty_mut(local.mutability), index, local.ty)
        } else {
            Ok(())
        }
    })?;

    body.var_debug_info.iter().try_for_each(|info| {
        let content = match &info.value {
            VarDebugInfoContents::Place(place) => {
                format!("{place:?}")
            }
            VarDebugInfoContents::Const(constant) => pretty_mir_const(&constant.const_),
        };
        writeln!(writer, "    debug {} => {};", info.name, content)
    })?;

    body.blocks
        .iter()
        .enumerate()
        .map(|(index, block)| -> io::Result<()> {
            writeln!(writer, "    bb{}: {{", index)?;
            let _ = block
                .statements
                .iter()
                .map(|statement| -> io::Result<()> {
                    pretty_statement(writer, &statement.kind)?;
                    Ok(())
                })
                .collect::<Vec<_>>();
            pretty_terminator(writer, &block.terminator.kind)?;
            writeln!(writer, "    }}").unwrap();
            Ok(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn pretty_statement<W: Write>(writer: &mut W, statement: &StatementKind) -> io::Result<()> {
    match statement {
        StatementKind::Assign(place, rval) => {
            write!(writer, "        {:?} = ", place)?;
            pretty_rvalue(writer, rval)?;
            writeln!(writer, ";")
        }
        // FIXME: Add rest of the statements
        StatementKind::FakeRead(cause, place) => {
            writeln!(writer, "FakeRead({cause:?}, {place:?});")
        }
        StatementKind::SetDiscriminant { place, variant_index } => {
            writeln!(writer, "discriminant({place:?} = {};", variant_index.to_index())
        }
        StatementKind::Deinit(place) => writeln!(writer, "Deinit({place:?};"),
        StatementKind::StorageLive(local) => {
            writeln!(writer, "StorageLive(_{local});")
        }
        StatementKind::StorageDead(local) => {
            writeln!(writer, "StorageDead(_{local});")
        }
        StatementKind::Retag(kind, place) => writeln!(writer, "Retag({kind:?}, {place:?});"),
        StatementKind::PlaceMention(place) => {
            writeln!(writer, "PlaceMention({place:?};")
        }
        StatementKind::ConstEvalCounter => {
            writeln!(writer, "ConstEvalCounter;")
        }
        StatementKind::Nop => writeln!(writer, "nop;"),
        StatementKind::AscribeUserType { .. }
        | StatementKind::Coverage(_)
        | StatementKind::Intrinsic(_) => {
            // FIX-ME: Make them pretty.
            writeln!(writer, "{statement:?};")
        }
    }
}

fn pretty_terminator<W: Write>(writer: &mut W, terminator: &TerminatorKind) -> io::Result<()> {
    pretty_terminator_head(writer, terminator)?;
    let successors = terminator.successors();
    let successor_count = successors.len();
    let labels = pretty_successor_labels(terminator);

    let show_unwind = !matches!(terminator.unwind(), None | Some(UnwindAction::Cleanup(_)));
    let fmt_unwind = |w: &mut W| -> io::Result<()> {
        write!(w, "unwind ")?;
        match terminator.unwind() {
            None | Some(UnwindAction::Cleanup(_)) => unreachable!(),
            Some(UnwindAction::Continue) => write!(w, "continue"),
            Some(UnwindAction::Unreachable) => write!(w, "unreachable"),
            Some(UnwindAction::Terminate) => write!(w, "terminate"),
        }
    };

    match (successor_count, show_unwind) {
        (0, false) => {}
        (0, true) => {
            write!(writer, " -> ")?;
            fmt_unwind(writer)?;
        }
        (1, false) => write!(writer, " -> bb{:?}", successors[0])?,
        _ => {
            write!(writer, " -> [")?;
            for (i, target) in successors.iter().enumerate() {
                if i > 0 {
                    write!(writer, ", ")?;
                }
                write!(writer, "{}: bb{:?}", labels[i], target)?;
            }
            if show_unwind {
                write!(writer, ", ")?;
                fmt_unwind(writer)?;
            }
            write!(writer, "]")?;
        }
    };

    writeln!(writer, ";")
}

fn pretty_terminator_head<W: Write>(writer: &mut W, terminator: &TerminatorKind) -> io::Result<()> {
    use self::TerminatorKind::*;
    const INDENT: &str = "        ";
    match terminator {
        Goto { .. } => write!(writer, "{INDENT}goto"),
        SwitchInt { discr, .. } => {
            write!(writer, "{INDENT}switchInt({})", pretty_operand(discr))
        }
        Resume => write!(writer, "{INDENT}resume"),
        Abort => write!(writer, "{INDENT}abort"),
        Return => write!(writer, "{INDENT}return"),
        Unreachable => write!(writer, "{INDENT}unreachable"),
        Drop { place, .. } => write!(writer, "{INDENT}drop({:?})", place),
        Call { func, args, destination, .. } => {
            write!(writer, "{INDENT}{:?} = {}(", destination, pretty_operand(func))?;
            let mut args_iter = args.iter();
            args_iter.next().map_or(Ok(()), |arg| write!(writer, "{}", pretty_operand(arg)))?;
            args_iter.try_for_each(|arg| write!(writer, ", {}", pretty_operand(arg)))?;
            write!(writer, ")")
        }
        Assert { cond, expected, msg, target: _, unwind: _ } => {
            write!(writer, "{INDENT}assert(")?;
            if !expected {
                write!(writer, "!")?;
            }
            write!(writer, "{}, ", pretty_operand(cond))?;
            pretty_assert_message(writer, msg)?;
            write!(writer, ")")
        }
        InlineAsm { .. } => write!(writer, "{INDENT}InlineAsm"),
    }
}

fn pretty_successor_labels(terminator: &TerminatorKind) -> Vec<String> {
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
        InlineAsm { destination: Some(_), .. } => vec!["goto".into(), "unwind".into()],
        InlineAsm { destination: None, .. } => vec!["unwind".into()],
    }
}

fn pretty_assert_message<W: Write>(writer: &mut W, msg: &AssertMessage) -> io::Result<()> {
    match msg {
        AssertMessage::BoundsCheck { len, index } => {
            let pretty_len = pretty_operand(len);
            let pretty_index = pretty_operand(index);
            write!(
                writer,
                "\"index out of bounds: the length is {{}} but the index is {{}}\", {pretty_len}, {pretty_index}"
            )
        }
        AssertMessage::Overflow(BinOp::Add, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            write!(
                writer,
                "\"attempt to compute `{{}} + {{}}`, which would overflow\", {pretty_l}, {pretty_r}"
            )
        }
        AssertMessage::Overflow(BinOp::Sub, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            write!(
                writer,
                "\"attempt to compute `{{}} - {{}}`, which would overflow\", {pretty_l}, {pretty_r}"
            )
        }
        AssertMessage::Overflow(BinOp::Mul, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            write!(
                writer,
                "\"attempt to compute `{{}} * {{}}`, which would overflow\", {pretty_l}, {pretty_r}"
            )
        }
        AssertMessage::Overflow(BinOp::Div, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            write!(
                writer,
                "\"attempt to compute `{{}} / {{}}`, which would overflow\", {pretty_l}, {pretty_r}"
            )
        }
        AssertMessage::Overflow(BinOp::Rem, l, r) => {
            let pretty_l = pretty_operand(l);
            let pretty_r = pretty_operand(r);
            write!(
                writer,
                "\"attempt to compute `{{}} % {{}}`, which would overflow\", {pretty_l}, {pretty_r}"
            )
        }
        AssertMessage::Overflow(BinOp::Shr, _, r) => {
            let pretty_r = pretty_operand(r);
            write!(writer, "\"attempt to shift right by `{{}}`, which would overflow\", {pretty_r}")
        }
        AssertMessage::Overflow(BinOp::Shl, _, r) => {
            let pretty_r = pretty_operand(r);
            write!(writer, "\"attempt to shift left by `{{}}`, which would overflow\", {pretty_r}")
        }
        AssertMessage::Overflow(op, _, _) => unreachable!("`{:?}` cannot overflow", op),
        AssertMessage::OverflowNeg(op) => {
            let pretty_op = pretty_operand(op);
            write!(writer, "\"attempt to negate `{{}}`, which would overflow\", {pretty_op}")
        }
        AssertMessage::DivisionByZero(op) => {
            let pretty_op = pretty_operand(op);
            write!(writer, "\"attempt to divide `{{}}` by zero\", {pretty_op}")
        }
        AssertMessage::RemainderByZero(op) => {
            let pretty_op = pretty_operand(op);
            write!(
                writer,
                "\"attempt to calculate the remainder of `{{}}` with a divisor of zero\", {pretty_op}"
            )
        }
        AssertMessage::MisalignedPointerDereference { required, found } => {
            let pretty_required = pretty_operand(required);
            let pretty_found = pretty_operand(found);
            write!(
                writer,
                "\"misaligned pointer dereference: address must be a multiple of {{}} but is {{}}\",{pretty_required}, {pretty_found}"
            )
        }
        AssertMessage::ResumedAfterReturn(_) | AssertMessage::ResumedAfterPanic(_) => {
            write!(writer, "{}", msg.description().unwrap())
        }
    }
}

fn pretty_operand(operand: &Operand) -> String {
    match operand {
        Operand::Copy(copy) => {
            format!("{:?}", copy)
        }
        Operand::Move(mv) => {
            format!("move {:?}", mv)
        }
        Operand::Constant(cnst) => pretty_mir_const(&cnst.const_),
    }
}

fn pretty_mir_const(literal: &MirConst) -> String {
    with(|cx| cx.mir_const_pretty(literal))
}

fn pretty_ty_const(ct: &TyConst) -> String {
    with(|cx| cx.ty_const_pretty(ct.id))
}

fn pretty_rvalue<W: Write>(writer: &mut W, rval: &Rvalue) -> io::Result<()> {
    match rval {
        Rvalue::AddressOf(mutability, place) => {
            write!(writer, "&raw {}(*{:?})", pretty_mut(*mutability), place)
        }
        Rvalue::Aggregate(aggregate_kind, operands) => {
            // FIXME: Add pretty_aggregate function that returns a pretty string
            write!(writer, "{aggregate_kind:?} (")?;
            let mut op_iter = operands.iter();
            op_iter.next().map_or(Ok(()), |op| write!(writer, "{}", pretty_operand(op)))?;
            op_iter.try_for_each(|op| write!(writer, ", {}", pretty_operand(op)))?;
            write!(writer, ")")
        }
        Rvalue::BinaryOp(bin, op1, op2) => {
            write!(writer, "{:?}({}, {})", bin, pretty_operand(op1), pretty_operand(op2))
        }
        Rvalue::Cast(_, op, ty) => {
            write!(writer, "{} as {}", pretty_operand(op), ty)
        }
        Rvalue::CheckedBinaryOp(bin, op1, op2) => {
            write!(writer, "Checked{:?}({}, {})", bin, pretty_operand(op1), pretty_operand(op2))
        }
        Rvalue::CopyForDeref(deref) => {
            write!(writer, "CopyForDeref({:?})", deref)
        }
        Rvalue::Discriminant(place) => {
            write!(writer, "discriminant({:?})", place)
        }
        Rvalue::Len(len) => {
            write!(writer, "len({:?})", len)
        }
        Rvalue::Ref(_, borrowkind, place) => {
            let kind = match borrowkind {
                BorrowKind::Shared => "&",
                BorrowKind::Fake(FakeBorrowKind::Deep) => "&fake ",
                BorrowKind::Fake(FakeBorrowKind::Shallow) => "&fake shallow ",
                BorrowKind::Mut { .. } => "&mut ",
            };
            write!(writer, "{kind}{:?}", place)
        }
        Rvalue::Repeat(op, cnst) => {
            write!(writer, "{} \" \" {}", pretty_operand(op), pretty_ty_const(cnst))
        }
        Rvalue::ShallowInitBox(_, _) => Ok(()),
        Rvalue::ThreadLocalRef(item) => {
            write!(writer, "thread_local_ref{:?}", item)
        }
        Rvalue::NullaryOp(nul, ty) => {
            write!(writer, "{:?} {} \" \"", nul, ty)
        }
        Rvalue::UnaryOp(un, op) => {
            write!(writer, "{} \" \" {:?}", pretty_operand(op), un)
        }
        Rvalue::Use(op) => write!(writer, "{}", pretty_operand(op)),
    }
}

fn pretty_mut(mutability: Mutability) -> &'static str {
    match mutability {
        Mutability::Not => " ",
        Mutability::Mut => "mut ",
    }
}
