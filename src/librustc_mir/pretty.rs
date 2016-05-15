// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{Location, ScopeAuxiliaryVec};
use rustc::hir;
use rustc::mir::repr::*;
use rustc::mir::transform::MirSource;
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::fnv::FnvHashMap;
use std::fmt::Display;
use std::fs;
use std::io::{self, Write};
use syntax::ast::NodeId;
use syntax::codemap::Span;

const INDENT: &'static str = "    ";
/// Alignment for lining up comments following MIR statements
const ALIGN: usize = 40;

/// If the session is properly configured, dumps a human-readable
/// representation of the mir into:
///
/// ```text
/// rustc.node<node_id>.<pass_name>.<disambiguator>
/// ```
///
/// Output from this function is controlled by passing `-Z dump-mir=<filter>`,
/// where `<filter>` takes the following forms:
///
/// - `all` -- dump MIR for all fns, all passes, all everything
/// - `substring1&substring2,...` -- `&`-separated list of substrings
///   that can appear in the pass-name or the `item_path_str` for the given
///   node-id. If any one of the substrings match, the data is dumped out.
pub fn dump_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          pass_name: &str,
                          disambiguator: &Display,
                          src: MirSource,
                          mir: &Mir<'tcx>,
                          auxiliary: Option<&ScopeAuxiliaryVec>) {
    let filters = match tcx.sess.opts.debugging_opts.dump_mir {
        None => return,
        Some(ref filters) => filters,
    };
    let node_id = src.item_id();
    let node_path = tcx.item_path_str(tcx.map.local_def_id(node_id));
    let is_matched =
        filters.split("&")
               .any(|filter| {
                   filter == "all" ||
                       pass_name.contains(filter) ||
                       node_path.contains(filter)
               });
    if !is_matched {
        return;
    }

    let file_name = format!("rustc.node{}.{}.{}.mir",
                            node_id, pass_name, disambiguator);
    let _ = fs::File::create(&file_name).and_then(|mut file| {
        try!(writeln!(file, "// MIR for `{}`", node_path));
        try!(writeln!(file, "// node_id = {}", node_id));
        try!(writeln!(file, "// pass_name = {}", pass_name));
        try!(writeln!(file, "// disambiguator = {}", disambiguator));
        try!(writeln!(file, ""));
        try!(write_mir_fn(tcx, src, mir, &mut file, auxiliary));
        Ok(())
    });
}

/// Write out a human-readable textual representation for the given MIR.
pub fn write_mir_pretty<'a, 'b, 'tcx, I>(tcx: TyCtxt<'b, 'tcx, 'tcx>,
                                         iter: I,
                                         w: &mut Write)
                                         -> io::Result<()>
    where I: Iterator<Item=(&'a NodeId, &'a Mir<'tcx>)>, 'tcx: 'a
{
    let mut first = true;
    for (&id, mir) in iter {
        if first {
            first = false;
        } else {
            // Put empty lines between all items
            writeln!(w, "")?;
        }

        let src = MirSource::from_node(tcx, id);
        write_mir_fn(tcx, src, mir, w, None)?;

        for (i, mir) in mir.promoted.iter().enumerate() {
            writeln!(w, "")?;
            write_mir_fn(tcx, MirSource::Promoted(id, i), mir, w, None)?;
        }
    }
    Ok(())
}

enum Annotation {
    EnterScope(ScopeId),
    ExitScope(ScopeId),
}

pub fn write_mir_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              src: MirSource,
                              mir: &Mir<'tcx>,
                              w: &mut Write,
                              auxiliary: Option<&ScopeAuxiliaryVec>)
                              -> io::Result<()> {
    // compute scope/entry exit annotations
    let mut annotations = FnvHashMap();
    if let Some(auxiliary) = auxiliary {
        for (index, auxiliary) in auxiliary.vec.iter().enumerate() {
            let scope_id = ScopeId::new(index);

            annotations.entry(auxiliary.dom)
                       .or_insert(vec![])
                       .push(Annotation::EnterScope(scope_id));

            for &loc in &auxiliary.postdoms {
                annotations.entry(loc)
                           .or_insert(vec![])
                           .push(Annotation::ExitScope(scope_id));
            }
        }
    }

    write_mir_intro(tcx, src, mir, w)?;
    for block in mir.all_basic_blocks() {
        write_basic_block(tcx, block, mir, w, &annotations)?;
    }

    // construct a scope tree and write it out
    let mut scope_tree: FnvHashMap<Option<ScopeId>, Vec<ScopeId>> = FnvHashMap();
    for (index, scope_data) in mir.scopes.iter().enumerate() {
        scope_tree.entry(scope_data.parent_scope)
                  .or_insert(vec![])
                  .push(ScopeId::new(index));
    }

    writeln!(w, "{}scope tree:", INDENT)?;
    write_scope_tree(tcx, mir, auxiliary, &scope_tree, w, None, 1, false)?;
    writeln!(w, "")?;

    writeln!(w, "}}")?;
    Ok(())
}

/// Write out a human-readable textual representation for the given basic block.
fn write_basic_block(tcx: TyCtxt,
                     block: BasicBlock,
                     mir: &Mir,
                     w: &mut Write,
                     annotations: &FnvHashMap<Location, Vec<Annotation>>)
                     -> io::Result<()> {
    let data = mir.basic_block_data(block);

    // Basic block label at the top.
    writeln!(w, "{}{:?}: {{", INDENT, block)?;

    // List of statements in the middle.
    let mut current_location = Location { block: block, statement_index: 0 };
    for statement in &data.statements {
        if let Some(ref annotations) = annotations.get(&current_location) {
            for annotation in annotations.iter() {
                match *annotation {
                    Annotation::EnterScope(id) =>
                        writeln!(w, "{0}{0}// Enter Scope({1})",
                                 INDENT, id.index())?,
                    Annotation::ExitScope(id) =>
                        writeln!(w, "{0}{0}// Exit Scope({1})",
                                 INDENT, id.index())?,
                }
            }
        }

        let indented_mir = format!("{0}{0}{1:?};", INDENT, statement);
        writeln!(w, "{0:1$} // {2}",
                 indented_mir,
                 ALIGN,
                 comment(tcx, statement.scope, statement.span))?;

        current_location.statement_index += 1;
    }

    // Terminator at the bottom.
    let indented_terminator = format!("{0}{0}{1:?};", INDENT, data.terminator().kind);
    writeln!(w, "{0:1$} // {2}",
             indented_terminator,
             ALIGN,
             comment(tcx, data.terminator().scope, data.terminator().span))?;

    writeln!(w, "{}}}\n", INDENT)
}

fn comment(tcx: TyCtxt, scope: ScopeId, span: Span) -> String {
    format!("scope {} at {}", scope.index(), tcx.sess.codemap().span_to_string(span))
}

fn write_scope_tree(tcx: TyCtxt,
                    mir: &Mir,
                    auxiliary: Option<&ScopeAuxiliaryVec>,
                    scope_tree: &FnvHashMap<Option<ScopeId>, Vec<ScopeId>>,
                    w: &mut Write,
                    parent: Option<ScopeId>,
                    depth: usize,
                    same_line: bool)
                    -> io::Result<()> {
    let indent = if same_line {
        0
    } else {
        depth * INDENT.len()
    };

    let children = match scope_tree.get(&parent) {
        Some(childs) => childs,
        None => return Ok(()),
    };

    for (index, &child) in children.iter().enumerate() {
        if index == 0 && same_line {
            // We know we're going to output a scope, so prefix it with a space to separate it from
            // the previous scopes on this line
            write!(w, " ")?;
        }

        let data = &mir.scopes[child];
        assert_eq!(data.parent_scope, parent);
        write!(w, "{0:1$}{2}", "", indent, child.index())?;

        let indent = indent + INDENT.len();

        if let Some(auxiliary) = auxiliary {
            let extent = auxiliary[child].extent;
            let data = tcx.region_maps.code_extent_data(extent);
            writeln!(w, "{0:1$}Extent: {2:?}", "", indent, data)?;
        }

        let child_count = scope_tree.get(&Some(child)).map(Vec::len).unwrap_or(0);
        if child_count < 2 {
            // Skip the braces when there's no or only a single subscope
            write_scope_tree(tcx, mir, auxiliary, scope_tree, w,
                             Some(child), depth, true)?;
        } else {
            // 2 or more child scopes? Put them in braces and on new lines.
            writeln!(w, " {{")?;
            write_scope_tree(tcx, mir, auxiliary, scope_tree, w,
                             Some(child), depth + 1, false)?;

            write!(w, "\n{0:1$}}}", "", depth * INDENT.len())?;
        }

        if !same_line && index + 1 < children.len() {
            writeln!(w, "")?;
        }
    }

    Ok(())
}

/// Write out a human-readable textual representation of the MIR's `fn` type and the types of its
/// local variables (both user-defined bindings and compiler temporaries).
fn write_mir_intro<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             src: MirSource,
                             mir: &Mir,
                             w: &mut Write)
                             -> io::Result<()> {
    match src {
        MirSource::Fn(_) => write!(w, "fn")?,
        MirSource::Const(_) => write!(w, "const")?,
        MirSource::Static(_, hir::MutImmutable) => write!(w, "static")?,
        MirSource::Static(_, hir::MutMutable) => write!(w, "static mut")?,
        MirSource::Promoted(_, i) => write!(w, "promoted{} in", i)?
    }

    write!(w, " {}", tcx.node_path_str(src.item_id()))?;

    if let MirSource::Fn(_) = src {
        write!(w, "(")?;

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
    } else {
        assert!(mir.arg_decls.is_empty());
        write!(w, ": {} =", mir.return_ty.unwrap())?;
    }

    writeln!(w, " {{")?;

    // User variable types (including the user's name in a comment).
    for (i, var) in mir.var_decls.iter().enumerate() {
        let mut_str = if var.mutability == Mutability::Mut {
            "mut "
        } else {
            ""
        };

        let indented_var = format!("{}let {}{:?}: {};",
                                   INDENT,
                                   mut_str,
                                   Lvalue::Var(i as u32),
                                   var.ty);
        writeln!(w, "{0:1$} // \"{2}\" in {3}",
                 indented_var,
                 ALIGN,
                 var.name,
                 comment(tcx, var.scope, var.span))?;
    }

    // Compiler-introduced temporary types.
    for (i, temp) in mir.temp_decls.iter().enumerate() {
        writeln!(w, "{}let mut {:?}: {};", INDENT, Lvalue::Temp(i as u32), temp.ty)?;
    }

    // Wrote any declaration? Add an empty line before the first block is printed.
    if !mir.var_decls.is_empty() || !mir.temp_decls.is_empty() {
        writeln!(w, "")?;
    }

    Ok(())
}
