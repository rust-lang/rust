// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{Location, ScopeAuxiliaryVec, ScopeId};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::mir::repr::*;
use rustc::mir::mir_map::MirMap;
use rustc::mir::transform::MirSource;
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::fnv::FnvHashMap;
use rustc_data_structures::indexed_vec::{Idx};
use std::fmt::Display;
use std::fs;
use std::io::{self, Write};
use std::path::{PathBuf, Path};

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

    let promotion_id = match src {
        MirSource::Promoted(_, id) => format!("-{:?}", id),
        _ => String::new()
    };

    let mut file_path = PathBuf::new();
    if let Some(ref file_dir) = tcx.sess.opts.debugging_opts.dump_mir_dir {
        let p = Path::new(file_dir);
        file_path.push(p);
    };
    let file_name = format!("rustc.node{}{}.{}.{}.mir",
                            node_id, promotion_id, pass_name, disambiguator);
    file_path.push(&file_name);
    let _ = fs::File::create(&file_path).and_then(|mut file| {
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
                                         mir_map: &MirMap<'tcx>,
                                         w: &mut Write)
                                         -> io::Result<()>
    where I: Iterator<Item=DefId>, 'tcx: 'a
{
    let mut first = true;
    for def_id in iter {
        let mir = &mir_map.map[&def_id];

        if first {
            first = false;
        } else {
            // Put empty lines between all items
            writeln!(w, "")?;
        }

        let id = tcx.map.as_local_node_id(def_id).unwrap();
        let src = MirSource::from_node(tcx, id);
        write_mir_fn(tcx, src, mir, w, None)?;

        for (i, mir) in mir.promoted.iter_enumerated() {
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

fn scope_entry_exit_annotations(auxiliary: Option<&ScopeAuxiliaryVec>)
                                -> FnvHashMap<Location, Vec<Annotation>>
{
    // compute scope/entry exit annotations
    let mut annotations = FnvHashMap();
    if let Some(auxiliary) = auxiliary {
        for (scope_id, auxiliary) in auxiliary.iter_enumerated() {
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
    return annotations;
}

pub fn write_mir_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              src: MirSource,
                              mir: &Mir<'tcx>,
                              w: &mut Write,
                              auxiliary: Option<&ScopeAuxiliaryVec>)
                              -> io::Result<()> {
    let annotations = scope_entry_exit_annotations(auxiliary);
    write_mir_intro(tcx, src, mir, w)?;
    for block in mir.basic_blocks().indices() {
        write_basic_block(tcx, block, mir, w, &annotations)?;
        if block.index() + 1 != mir.basic_blocks().len() {
            writeln!(w, "")?;
        }
    }

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
    let data = &mir[block];

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
                 comment(tcx, statement.source_info))?;

        current_location.statement_index += 1;
    }

    // Terminator at the bottom.
    let indented_terminator = format!("{0}{0}{1:?};", INDENT, data.terminator().kind);
    writeln!(w, "{0:1$} // {2}",
             indented_terminator,
             ALIGN,
             comment(tcx, data.terminator().source_info))?;

    writeln!(w, "{}}}", INDENT)
}

fn comment(tcx: TyCtxt, SourceInfo { span, scope }: SourceInfo) -> String {
    format!("scope {} at {}", scope.index(), tcx.sess.codemap().span_to_string(span))
}

fn write_scope_tree(tcx: TyCtxt,
                    mir: &Mir,
                    scope_tree: &FnvHashMap<VisibilityScope, Vec<VisibilityScope>>,
                    w: &mut Write,
                    parent: VisibilityScope,
                    depth: usize)
                    -> io::Result<()> {
    let indent = depth * INDENT.len();

    let children = match scope_tree.get(&parent) {
        Some(childs) => childs,
        None => return Ok(()),
    };

    for &child in children {
        let data = &mir.visibility_scopes[child];
        assert_eq!(data.parent_scope, Some(parent));
        writeln!(w, "{0:1$}scope {2} {{", "", indent, child.index())?;

        // User variable types (including the user's name in a comment).
        for (id, var) in mir.var_decls.iter_enumerated() {
            // Skip if not declared in this scope.
            if var.source_info.scope != child {
                continue;
            }

            let mut_str = if var.mutability == Mutability::Mut {
                "mut "
            } else {
                ""
            };

            let indent = indent + INDENT.len();
            let indented_var = format!("{0:1$}let {2}{3:?}: {4};",
                                       INDENT,
                                       indent,
                                       mut_str,
                                       id,
                                       var.ty);
            writeln!(w, "{0:1$} // \"{2}\" in {3}",
                     indented_var,
                     ALIGN,
                     var.name,
                     comment(tcx, var.source_info))?;
        }

        write_scope_tree(tcx, mir, scope_tree, w, child, depth + 1)?;

        writeln!(w, "{0:1$}}}", "", depth * INDENT.len())?;
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
    write_mir_sig(tcx, src, mir, w)?;
    writeln!(w, " {{")?;

    // construct a scope tree and write it out
    let mut scope_tree: FnvHashMap<VisibilityScope, Vec<VisibilityScope>> = FnvHashMap();
    for (index, scope_data) in mir.visibility_scopes.iter().enumerate() {
        if let Some(parent) = scope_data.parent_scope {
            scope_tree.entry(parent)
                      .or_insert(vec![])
                      .push(VisibilityScope::new(index));
        } else {
            // Only the argument scope has no parent, because it's the root.
            assert_eq!(index, ARGUMENT_VISIBILITY_SCOPE.index());
        }
    }

    write_scope_tree(tcx, mir, &scope_tree, w, ARGUMENT_VISIBILITY_SCOPE, 1)?;

    write_mir_decls(mir, w)
}

fn write_mir_sig(tcx: TyCtxt, src: MirSource, mir: &Mir, w: &mut Write)
                 -> io::Result<()>
{
    match src {
        MirSource::Fn(_) => write!(w, "fn")?,
        MirSource::Const(_) => write!(w, "const")?,
        MirSource::Static(_, hir::MutImmutable) => write!(w, "static")?,
        MirSource::Static(_, hir::MutMutable) => write!(w, "static mut")?,
        MirSource::Promoted(_, i) => write!(w, "{:?} in", i)?
    }

    write!(w, " {}", tcx.node_path_str(src.item_id()))?;

    if let MirSource::Fn(_) = src {
        write!(w, "(")?;

        // fn argument types.
        for (i, arg) in mir.arg_decls.iter_enumerated() {
            if i.index() != 0 {
                write!(w, ", ")?;
            }
            write!(w, "{:?}: {}", Lvalue::Arg(i), arg.ty)?;
        }

        write!(w, ") -> ")?;

        // fn return type.
        match mir.return_ty {
            ty::FnOutput::FnConverging(ty) => write!(w, "{}", ty),
            ty::FnOutput::FnDiverging => write!(w, "!"),
        }
    } else {
        assert!(mir.arg_decls.is_empty());
        write!(w, ": {} =", mir.return_ty.unwrap())
    }
}

fn write_mir_decls(mir: &Mir, w: &mut Write) -> io::Result<()> {
    // Compiler-introduced temporary types.
    for (id, temp) in mir.temp_decls.iter_enumerated() {
        writeln!(w, "{}let mut {:?}: {};", INDENT, id, temp.ty)?;
    }

    // Wrote any declaration? Add an empty line before the first block is printed.
    if !mir.var_decls.is_empty() || !mir.temp_decls.is_empty() {
        writeln!(w, "")?;
    }

    Ok(())
}
