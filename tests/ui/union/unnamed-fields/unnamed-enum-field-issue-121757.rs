type NodeId = u32;
struct Type<'a>(std::marker::PhantomData::<&'a ()>);

type Ast<'ast> = &'ast AstStructure<'ast>;

struct AstStructure<'ast> {
//~^ ERROR struct with unnamed fields must have `#[repr(C)]` representation
    id: NodeId,
    _: AstKind<'ast>
//~^ ERROR unnamed fields are not yet fully implemented [E0658]
//~^^ ERROR unnamed fields can only have struct or union types
}

enum AstKind<'ast> {
    ExprInt,
    ExprLambda(Ast<'ast>),
}

fn compute_types<'tcx,'ast>(ast: Ast<'ast>) -> Type<'tcx>
{
    match ast.kind {}
//~^ ERROR no field `kind` on type `&'ast AstStructure<'ast>` [E0609]
}

fn main() {}
