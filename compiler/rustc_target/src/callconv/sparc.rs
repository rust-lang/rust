use rustc_abi::{
    BackendRepr, CanonAbi, Float, HasDataLayout, Integer, Numeric, Primitive, RegKind,
    TyAbiInterface,
};

use crate::callconv::{ArgAbi, ArgAttribute, CastTarget, ComputeCallConv, Reg};

fn classify_shared<'a, Ty>(val: &mut ArgAbi<'a, Ty>) {
    if val.layout.is_aggregate() {
        val.make_indirect();
    } else if let BackendRepr::Scalar(scalar) = val.layout.backend_repr
        && scalar.primitive() == Primitive::Float(Float::F128)
    {
        // f128 is passed and returned indirectly.
        val.make_indirect();
    } else {
        val.extend_integer_width_to(32);
    }
}

fn classify_complex_ret<'a, Ty>(ret: &mut ArgAbi<'a, Ty>, component: Numeric) {
    let reg = Reg { kind: component.reg_kind(), size: component.size() };
    let mut cast = CastTarget::pair(reg, reg);

    match component {
        Numeric::Float(Float::F128) => {
            // Mark `_Complex long double` as inreg to get the right behavior,
            // consistent with clang `SparcV8ABIInfo::classifyReturnType`.
            cast.attrs.set(ArgAttribute::InReg);
        }
        Numeric::Int(Integer::I8 | Integer::I16 | Integer::I32, _) => {
            let size = ret.layout.size;
            cast = CastTarget::from(Reg { kind: RegKind::Integer, size });
        }
        _ => { /* default behavior */ }
    }

    ret.cast_to(cast);
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }

    if let Some(component) = arg.layout.complex_number(cx) {
        if let Numeric::Int(Integer::I8 | Integer::I16 | Integer::I32, _) = component {
            arg.cast_to(Reg { kind: RegKind::Integer, size: 2 * component.size() });
        } else {
            arg.make_indirect();
        }
        return;
    }

    classify_shared(arg)
}

pub(crate) struct SparcCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for SparcCallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn passes_zst(&self) -> bool {
        true
    }

    fn compute_ret_abi(&mut self, cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if !ret.is_ignore() {
            if let Some(component) = ret.layout.complex_number(cx) {
                classify_complex_ret(ret, component);
            } else {
                classify_shared(ret);
            };
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            arg.make_indirect_from_ignore();
            return;
        }
        classify_arg(cx, arg);
    }
}
