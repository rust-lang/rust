// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::repr::*;
use rustc::middle::ty;
use std::io::{self, Write};
use syntax::ast::NodeId;

const INDENT: &'static str = "    ";

/// Write out a human-readable textual representation for the given MIR.
pub fn write_mir_pretty<'a, 't, W, I>(tcx: &ty::TyCtxt<'t>, iter: I, w: &mut W) -> io::Result<()>
where W: Write, I: Iterator<Item=(&'a NodeId, &'a Mir<'a>)> {
    for (&nodeid, mir) in iter {
        write_mir_intro(tcx, nodeid, mir, w)?;
        // Nodes
        for block in mir.all_basic_blocks() {
            write_basic_block(block, mir, w)?;
        }
        writeln!(w, "}}")?
    }
    Ok(())
}

/// Write out a human-readable textual representation for the given basic block.
fn write_basic_block<W: Write>(block: BasicBlock, mir: &Mir, w: &mut W) -> io::Result<()> {
    let data = mir.basic_block_data(block);

    // Basic block label at the top.
    writeln!(w, "\n{}{:?}: {{", INDENT, block)?;

    // List of statements in the middle.
    for statement in &data.statements {
        writeln!(w, "{0}{0}{1:?};", INDENT, statement)?;
    }

    // Terminator at the bottom.
    writeln!(w, "{0}{0}{1:?};", INDENT, data.terminator())?;

    writeln!(w, "{}}}", INDENT)
}

/// Write out a human-readable textual representation of the MIR's `fn` type and the types of its
/// local variables (both user-defined bindings and compiler temporaries).
fn write_mir_intro<W: Write>(tcx: &ty::TyCtxt, nid: NodeId, mir: &Mir, w: &mut W)
-> io::Result<()> {

    write!(w, "fn {}(", tcx.map.path_to_string(nid))?;

    // fn argument types.
    for (i, arg) in mir.arg_decls.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:?}: {}", Lvalue::Arg(i as u32), arg.ty)?;
    }

    write!(w, ") -> ")?;

    // fn return type.
    match mir.return_ty {
        ty::FnOutput::FnConverging(ty) => write!(w, "{}", ty)?,
        ty::FnOutput::FnDiverging => write!(w, "!")?,
    }

    writeln!(w, " {{")?;

    // User variable types (including the user's name in a comment).
    for (i, var) in mir.var_decls.iter().enumerate() {
        write!(w, "{}let ", INDENT)?;
        if var.mutability == Mutability::Mut {
            write!(w, "mut ")?;
        }
        writeln!(w, "{:?}: {}; // {}", Lvalue::Var(i as u32), var.ty, var.name)?;
    }

    // Compiler-introduced temporary types.
    for (i, temp) in mir.temp_decls.iter().enumerate() {
        writeln!(w, "{}let mut {:?}: {};", INDENT, Lvalue::Temp(i as u32), temp.ty)?;
    }

    Ok(())
}
