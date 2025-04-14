//@ run-pass
//! Sanity check Stable MIR Visitor

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use stable_mir::mir::MirVisitor;
use stable_mir::mir::MutMirVisitor;
use stable_mir::*;
use std::collections::HashSet;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn test_visitor() -> ControlFlow<()> {
    let main_fn = stable_mir::entry_fn();
    let main_body = main_fn.unwrap().expect_body();
    let main_visitor = TestVisitor::collect(&main_body);
    assert!(main_visitor.ret_val.is_some());
    assert!(main_visitor.args.is_empty());
    assert!(main_visitor.tys.contains(&main_visitor.ret_val.unwrap().ty));
    assert!(!main_visitor.calls.is_empty());

    let exit_fn = main_visitor.calls.last().unwrap();
    assert!(exit_fn.mangled_name().contains("exit_fn"), "Unexpected last function: {exit_fn:?}");

    let exit_body = exit_fn.body().unwrap();
    let exit_visitor = TestVisitor::collect(&exit_body);
    assert!(exit_visitor.ret_val.is_some());
    assert_eq!(exit_visitor.args.len(), 1);
    assert!(exit_visitor.tys.contains(&exit_visitor.ret_val.unwrap().ty));
    assert!(exit_visitor.tys.contains(&exit_visitor.args[0].ty));
    ControlFlow::Continue(())
}

struct TestVisitor<'a> {
    pub body: &'a mir::Body,
    pub tys: HashSet<ty::Ty>,
    pub ret_val: Option<mir::LocalDecl>,
    pub args: Vec<mir::LocalDecl>,
    pub calls: Vec<mir::mono::Instance>,
}

impl<'a> TestVisitor<'a> {
    fn collect(body: &'a mir::Body) -> TestVisitor<'a> {
        let mut visitor = TestVisitor {
            body: &body,
            tys: Default::default(),
            ret_val: None,
            args: vec![],
            calls: vec![],
        };
        visitor.visit_body(&body);
        visitor
    }
}

impl<'a> mir::MirVisitor for TestVisitor<'a> {
    fn visit_ty(&mut self, ty: &ty::Ty, _location: mir::visit::Location) {
        self.tys.insert(*ty);
        self.super_ty(ty)
    }

    fn visit_ret_decl(&mut self, local: mir::Local, decl: &mir::LocalDecl) {
        assert!(local == mir::RETURN_LOCAL);
        assert!(self.ret_val.is_none());
        self.ret_val = Some(decl.clone());
        self.super_ret_decl(local, decl);
    }

    fn visit_arg_decl(&mut self, local: mir::Local, decl: &mir::LocalDecl) {
        self.args.push(decl.clone());
        assert_eq!(local, self.args.len());
        self.super_arg_decl(local, decl);
    }

    fn visit_terminator(&mut self, term: &mir::Terminator, location: mir::visit::Location) {
        if let mir::TerminatorKind::Call { func, .. } = &term.kind {
            let ty::TyKind::RigidTy(ty) = func.ty(self.body.locals()).unwrap().kind() else {
                unreachable!()
            };
            let ty::RigidTy::FnDef(def, args) = ty else { unreachable!() };
            self.calls.push(mir::mono::Instance::resolve(def, &args).unwrap());
        }
        self.super_terminator(term, location);
    }
}

fn test_mut_visitor() -> ControlFlow<()> {
    let main_fn = stable_mir::entry_fn();
    let mut main_body = main_fn.unwrap().expect_body();
    let locals = main_body.locals().to_vec();
    let mut main_visitor = TestMutVisitor::collect(locals);
    main_visitor.visit_body(&mut main_body);
    assert!(main_visitor.ret_val.is_some());
    assert!(main_visitor.args.is_empty());
    assert!(main_visitor.tys.contains(&main_visitor.ret_val.unwrap().ty));
    assert!(!main_visitor.calls.is_empty());

    let exit_fn = main_visitor.calls.last().unwrap();
    assert!(exit_fn.mangled_name().contains("exit_fn"), "Unexpected last function: {exit_fn:?}");

    let mut exit_body = exit_fn.body().unwrap();
    let locals = exit_body.locals().to_vec();
    let mut exit_visitor = TestMutVisitor::collect(locals);
    exit_visitor.visit_body(&mut exit_body);
    assert!(exit_visitor.ret_val.is_some());
    assert_eq!(exit_visitor.args.len(), 1);
    assert!(exit_visitor.tys.contains(&exit_visitor.ret_val.unwrap().ty));
    assert!(exit_visitor.tys.contains(&exit_visitor.args[0].ty));
    ControlFlow::Continue(())
}

struct TestMutVisitor {
    locals: Vec<mir::LocalDecl>,
    pub tys: HashSet<ty::Ty>,
    pub ret_val: Option<mir::LocalDecl>,
    pub args: Vec<mir::LocalDecl>,
    pub calls: Vec<mir::mono::Instance>,
}

impl TestMutVisitor {
    fn collect(locals: Vec<mir::LocalDecl>) -> TestMutVisitor {
        let visitor = TestMutVisitor {
            locals: locals,
            tys: Default::default(),
            ret_val: None,
            args: vec![],
            calls: vec![],
        };
        visitor
    }
}

impl mir::MutMirVisitor for TestMutVisitor {
    fn visit_ty(&mut self, ty: &mut ty::Ty, _location: mir::visit::Location) {
        self.tys.insert(*ty);
        self.super_ty(ty)
    }

    fn visit_ret_decl(&mut self, local: mir::Local, decl: &mut mir::LocalDecl) {
        assert!(local == mir::RETURN_LOCAL);
        assert!(self.ret_val.is_none());
        self.ret_val = Some(decl.clone());
        self.super_ret_decl(local, decl);
    }

    fn visit_arg_decl(&mut self, local: mir::Local, decl: &mut mir::LocalDecl) {
        self.args.push(decl.clone());
        assert_eq!(local, self.args.len());
        self.super_arg_decl(local, decl);
    }

    fn visit_terminator(&mut self, term: &mut mir::Terminator, location: mir::visit::Location) {
        if let mir::TerminatorKind::Call { func, .. } = &mut term.kind {
            let ty::TyKind::RigidTy(ty) = func.ty(&self.locals).unwrap().kind() else {
                unreachable!()
            };
            let ty::RigidTy::FnDef(def, args) = ty else { unreachable!() };
            self.calls.push(mir::mono::Instance::resolve(def, &args).unwrap());
        }
        self.super_terminator(term, location);
    }
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "sim_visitor_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args.clone(), test_visitor).unwrap();
    run!(args, test_mut_visitor).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    fn main() -> std::process::ExitCode {{
        let inputs = Inputs::new();
        let total = inputs.values.iter().sum();
        exit_fn(total)
    }}

    fn exit_fn(code: u8) -> std::process::ExitCode {{
        std::process::ExitCode::from(code)
    }}

    struct Inputs {{
        values: [u8; 3],
    }}

    impl Inputs {{
        fn new() -> Inputs {{
            Inputs {{ values: [0, 1, 2] }}
        }}
    }}
    "#
    )?;
    Ok(())
}
