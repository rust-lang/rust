//@ run-pass

#![allow(dead_code)]
#![allow(unused_imports)]

// Test a sample usage pattern for regions. Makes use of the
// following features:
//
// - Multiple lifetime parameters
// - Arenas

#![feature(rustc_private)]

extern crate rustc_arena;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use TypeStructure::{TypeInt, TypeFunction};
use AstKind::{ExprInt, ExprVar, ExprLambda};
use rustc_arena::TypedArena;
use std::collections::HashMap;
use std::mem;

type Type<'tcx> = &'tcx TypeStructure<'tcx>;

#[derive(Copy, Clone, Debug)]
enum TypeStructure<'tcx> {
    TypeInt,
    TypeFunction(Type<'tcx>, Type<'tcx>),
}

impl<'tcx> PartialEq for TypeStructure<'tcx> {
    fn eq(&self, other: &TypeStructure<'tcx>) -> bool {
        match (*self, *other) {
            (TypeInt, TypeInt) => true,
            (TypeFunction(s_a, s_b), TypeFunction(o_a, o_b)) => *s_a == *o_a && *s_b == *o_b,
            _ => false
        }
    }
}

impl<'tcx> Eq for TypeStructure<'tcx> {}

type TyArena<'tcx> = TypedArena<TypeStructure<'tcx>>;
type AstArena<'ast> = TypedArena<AstStructure<'ast>>;

struct TypeContext<'tcx, 'ast> {
    ty_arena: &'tcx TyArena<'tcx>,
    types: Vec<Type<'tcx>> ,
    type_table: HashMap<NodeId, Type<'tcx>>,

    ast_arena: &'ast AstArena<'ast>,
    ast_counter: usize,
}

impl<'tcx,'ast> TypeContext<'tcx, 'ast> {
    fn new(ty_arena: &'tcx TyArena<'tcx>, ast_arena: &'ast AstArena<'ast>)
           -> TypeContext<'tcx, 'ast> {
        TypeContext { ty_arena: ty_arena,
                      types: Vec::new(),
                      type_table: HashMap::new(),

                      ast_arena: ast_arena,
                      ast_counter: 0 }
    }

    fn add_type(&mut self, s: TypeStructure<'tcx>) -> Type<'tcx> {
        for &ty in &self.types {
            if *ty == s {
                return ty;
            }
        }

        let ty = self.ty_arena.alloc(s);
        self.types.push(ty);
        ty
    }

    fn set_type(&mut self, id: NodeId, ty: Type<'tcx>) -> Type<'tcx> {
        self.type_table.insert(id, ty);
        ty
    }

    fn ast(&mut self, a: AstKind<'ast>) -> Ast<'ast> {
        let id = self.ast_counter;
        self.ast_counter += 1;
        self.ast_arena.alloc(AstStructure { id: NodeId {id:id}, kind: a })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct NodeId {
    id: usize
}

type Ast<'ast> = &'ast AstStructure<'ast>;

#[derive(Copy, Clone)]
struct AstStructure<'ast> {
    id: NodeId,
    kind: AstKind<'ast>
}

#[derive(Copy, Clone)]
enum AstKind<'ast> {
    ExprInt,
    ExprVar(usize),
    ExprLambda(Ast<'ast>),
}

fn compute_types<'tcx,'ast>(tcx: &mut TypeContext<'tcx,'ast>,
                            ast: Ast<'ast>) -> Type<'tcx>
{
    match ast.kind {
        ExprInt | ExprVar(_) => {
            let ty = tcx.add_type(TypeInt);
            tcx.set_type(ast.id, ty)
        }
        ExprLambda(ast) => {
            let arg_ty = tcx.add_type(TypeInt);
            let body_ty = compute_types(tcx, ast);
            let lambda_ty = tcx.add_type(TypeFunction(arg_ty, body_ty));
            tcx.set_type(ast.id, lambda_ty)
        }
    }
}

pub fn main() {
    let ty_arena = TypedArena::default();
    let ast_arena = TypedArena::default();
    let mut tcx = TypeContext::new(&ty_arena, &ast_arena);
    let ast = tcx.ast(ExprInt);
    let ty = compute_types(&mut tcx, ast);
    assert_eq!(*ty, TypeInt);
}
