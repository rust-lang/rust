use std::fmt;
use std::iter;
use rustc_middle::ty::{self, Ty, TyCtxt, Adt, ParamEnv, ParamEnvAnd};
use rustc_target::abi::FieldsShape;

#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum Kind {
    Anything,
    Integer,
    Pointer,
    Half,
    Float,
    Double,
    Unknown,
}

#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct TypeTree(pub Vec<Type>);

#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct Type {
    pub offset: isize,
    pub size: usize,
    pub kind: Kind,
    pub child: TypeTree,
}

impl Type {
    pub fn add_offset(self, add: isize) -> Self {
        Self {
            offset: self.offset + add,
            size: self.size,
            kind: self.kind,
            child: self.child,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}

impl TypeTree {
    pub fn empty() -> Self {
        TypeTree(vec![])
    }

    pub fn from_ty<'a>(ty: Ty<'a>, tcx: TyCtxt<'a>, depth: usize) -> Self {
        if ty.is_scalar() {
            assert!(!ty.is_any_ptr());

            let (kind, size) = if ty.is_integral() {
                (Kind::Integer, 1)
            } else {
                assert!(ty.is_floating_point());
                match ty {
                    x if x == tcx.types.f32 => (Kind::Float, 4),
                    x if x == tcx.types.f64 => (Kind::Double, 8),
                    _ => panic!("floatTy scalar that is neither f32 nor f64"),
                }
            };

            return TypeTree(vec![
                Type {offset: -1, child: TypeTree::empty(), kind, size }
            ]);
        }

        if ty.is_unsafe_ptr() || ty.is_ref() || ty.is_box() {
            if  ty.is_fn_ptr() {
                unimplemented!("what to do whith fn ptr?");
            }

            let inner_ty = ty.builtin_deref(true).unwrap().ty;
            let child = TypeTree::from_ty(inner_ty, tcx, depth + 1);

            let tt = Type { offset: -1, kind: Kind::Pointer, size: 8, child, };
            println!("{:depth$} add indirection {:?}", "", tt);

            return TypeTree(vec![tt]);
        }

        let param_env_and = ParamEnvAnd {
            param_env: ParamEnv::empty(),
            value: ty,
        };

        let layout = tcx.layout_of(param_env_and);
        assert!(layout.is_ok());

        let layout = layout.unwrap().layout;
        let fields = layout.fields();
        let max_size = layout.size();

        if ty.is_adt() {
            let adt_def = ty.ty_adt_def().unwrap();
            let substs = match ty.kind() {
                Adt(_, subst_ref) => subst_ref,
                _ => panic!(""),
            };

            if adt_def.is_struct() {
                let (offsets, _memory_index) = match fields {
                    FieldsShape::Arbitrary{ offsets: o, memory_index: m } => (o,m),
                    _ => panic!(""),
                };
                println!("{:depth$} combine fields", "");

                let fields = adt_def.all_fields();
                let fields = fields.into_iter().zip(offsets.into_iter()).filter_map(|(field, offset)| {
                    let field_ty: Ty<'_> = field.ty(tcx, substs);
                    let field_ty: Ty<'_> = tcx.normalize_erasing_regions(ParamEnv::empty(), field_ty);

                    if field_ty.is_phantom_data() {
                        return None;
                    }

                    let mut child = TypeTree::from_ty(field_ty, tcx, depth + 1).0;

                    for c in &mut child {
                        if c.offset == -1 {
                            c.offset = offset.bytes() as isize
                        } else {
                            c.offset += offset.bytes() as isize;
                        }
                    }

                    //inner_tt.offset = offset;

                    println!("{:depth$} -> {:?}", "", child);

                    Some(child)
                }).flatten().collect::<Vec<Type>>();

                let ret_tt = TypeTree(fields);
                println!("{:depth$} into {:?}", "", ret_tt);
                return ret_tt;
            } else {
                unimplemented!("adt that isn't a struct");
            }
        }

        if ty.is_array() {
            let (stride, count) = match fields {
                FieldsShape::Array{ stride: s, count: c } => (s,c),
                _ => panic!(""),
            };
            let byte_stride = stride.bytes_usize();
            let byte_max_size = max_size.bytes_usize();

            assert!(byte_stride * *count as usize == byte_max_size);
            assert!(*count > 0); // return empty TT for empty?
            let sub_ty = ty.builtin_index().unwrap();
            let subtt = TypeTree::from_ty(sub_ty, tcx, depth + 1);

            // calculate size of subtree
            let param_env_and = ParamEnvAnd {
                param_env: ParamEnv::empty(),
                value: sub_ty,
            };
            let size = tcx.layout_of(param_env_and).unwrap().size.bytes() as usize;
            let tt = TypeTree(
                iter::repeat(subtt).take(*count as usize).enumerate().map(|(idx, x)| 
                    x.0.into_iter().map(move |x| x.add_offset((idx * size) as isize))
                ).flatten().collect());

            println!("{:depth$} repeated array into {:?}", "", tt);

            return tt;
        }

        println!("Warning: create empty typetree for {}", ty);
        TypeTree::empty()
    }
}

pub fn fnc_typetrees<'tcx>(fn_ty: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> (Vec<TypeTree>, TypeTree) {
    let fnc_binder: ty::Binder<'_, ty::FnSig<'_>> = fn_ty.fn_sig(tcx);

    // TODO: verify.
    let x: ty::FnSig<'_> = fnc_binder.skip_binder();

    let inputs = x.inputs().into_iter().map(|x| TypeTree::from_ty(*x, tcx, 0))
        .collect();

    let output = TypeTree::from_ty(x.output(), tcx, 0);

    (inputs, output)
}
