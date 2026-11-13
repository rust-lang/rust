// Tests that we don't fail with an overflow error for certain
// strange types
// See https://github.com/rust-lang/rust/pull/72936#issuecomment-643676915

pub trait Interner {
    type InternedType;
}

struct RustInterner<'tcx> {
    foo: &'tcx ()
}

impl<'tcx> Interner for RustInterner<'tcx> {
    type InternedType = Box<TyData<Self>>;
}

enum TyData<I: Interner> {
    FnDef(I::InternedType)
}

struct VariableKind<I: Interner>(I::InternedType);

//@ has overflow/struct.BoundVarsCollector.html
//@ has - '//h3[@class="code-header"]' "impl<'tcx> Send for BoundVarsCollector<'tcx>"
pub struct BoundVarsCollector<'tcx> {
    val: VariableKind<RustInterner<'tcx>>
}

fn is_send<T: Send>() {}

struct MyInterner<'tcx> {
    val: &'tcx ()
}

fn main() {}
