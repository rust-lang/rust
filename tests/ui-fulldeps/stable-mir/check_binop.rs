//@ run-pass
//! Test information regarding binary operations.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]

extern crate rustc_hir;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use stable_mir::mir::mono::Instance;
use stable_mir::mir::visit::{Location, MirVisitor};
use stable_mir::mir::{LocalDecl, Rvalue, Statement, StatementKind, Terminator, TerminatorKind};
use stable_mir::ty::{RigidTy, TyKind};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::io::Write;
use std::ops::ControlFlow;

/// This function tests that we can correctly get type information from binary operations.
fn test_binops() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = stable_mir::all_local_items();
    let mut instances =
        items.into_iter().map(|item| Instance::try_from(item).unwrap()).collect::<Vec<_>>();
    while let Some(instance) = instances.pop() {
        // The test below shouldn't have recursion in it.
        let Some(body) = instance.body() else {
            continue;
        };
        let mut visitor = Visitor { locals: body.locals(), calls: Default::default() };
        visitor.visit_body(&body);
        instances.extend(visitor.calls.into_iter());
    }
    ControlFlow::Continue(())
}

struct Visitor<'a> {
    locals: &'a [LocalDecl],
    calls: HashSet<Instance>,
}

impl<'a> MirVisitor for Visitor<'a> {
    fn visit_statement(&mut self, stmt: &Statement, _loc: Location) {
        match &stmt.kind {
            StatementKind::Assign(place, Rvalue::BinaryOp(op, rhs, lhs)) => {
                let ret_ty = place.ty(self.locals).unwrap();
                let op_ty = op.ty(rhs.ty(self.locals).unwrap(), lhs.ty(self.locals).unwrap());
                assert_eq!(ret_ty, op_ty, "Operation type should match the assigned place type");
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, term: &Terminator, _loc: Location) {
        match &term.kind {
            TerminatorKind::Call { func, .. } => {
                let TyKind::RigidTy(RigidTy::FnDef(def, args)) =
                    func.ty(self.locals).unwrap().kind()
                    else {
                        return;
                    };
                self.calls.insert(Instance::resolve(def, &args).unwrap());
            }
            _ => {}
        }
    }
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "binop_input.rs";
    generate_input(&path).unwrap();
    let args = &["rustc".to_string(), "--crate-type=lib".to_string(), path.to_string()];
    run!(args, test_binops).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        macro_rules! binop_int {{
            ($fn:ident, $typ:ty) => {{
                pub fn $fn(lhs: $typ, rhs: $typ) {{
                    let eq = lhs == rhs;
                    let lt = lhs < rhs;
                    let le = lhs <= rhs;

                    let sum = lhs + rhs;
                    let mult = lhs * sum;
                    let shift = mult << 2;
                    let bit_or = shift | rhs;
                    let cmp = lhs.cmp(&bit_or);

                    // Try to avoid the results above being pruned
                    std::hint::black_box(((eq, lt, le), cmp));
                }}
            }}
        }}

        binop_int!(binop_u8, u8);
        binop_int!(binop_i64, i64);

        pub fn binop_bool(lhs: bool, rhs: bool) {{
            let eq = lhs == rhs;
            let or = lhs | eq;
            let lt = lhs < or;
            let cmp = lhs.cmp(&rhs);

            // Try to avoid the results above being pruned
            std::hint::black_box((lt, cmp));
        }}

        pub fn binop_char(lhs: char, rhs: char) {{
            let eq = lhs == rhs;
            let lt = lhs < rhs;
            let cmp = lhs.cmp(&rhs);

            // Try to avoid the results above being pruned
            std::hint::black_box(([eq, lt], cmp));
        }}

        pub fn binop_ptr(lhs: *const char, rhs: *const char) {{
            let eq = lhs == rhs;
            let lt = lhs < rhs;
            let cmp = lhs.cmp(&rhs);
            let off = unsafe {{ lhs.offset(2) }};

            // Try to avoid the results above being pruned
            std::hint::black_box(([eq, lt], cmp, off));
        }}
        "#
    )?;
    Ok(())
}
