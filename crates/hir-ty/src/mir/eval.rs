//! This module provides a MIR interpreter, which is used in const eval.

use std::{borrow::Cow, collections::HashMap, iter, ops::Range, sync::Arc};

use base_db::CrateId;
use chalk_ir::{
    fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable},
    DebruijnIndex, TyKind,
};
use hir_def::{
    builtin_type::BuiltinType,
    lang_item::{lang_attr, LangItem},
    layout::{Layout, LayoutError, RustcEnumVariantIdx, TagEncoding, Variants},
    AdtId, DefWithBodyId, EnumVariantId, FunctionId, HasModule, ItemContainerId, Lookup, VariantId,
};
use intern::Interned;
use la_arena::ArenaMap;

use crate::{
    consteval::{intern_const_scalar, ConstEvalError},
    db::HirDatabase,
    from_placeholder_idx,
    infer::{normalize, PointerCast},
    layout::layout_of_ty,
    mapping::from_chalk,
    method_resolution::{is_dyn_method, lookup_impl_method},
    traits::FnTrait,
    CallableDefId, ClosureId, Const, ConstScalar, FnDefId, GenericArgData, Interner, MemoryMap,
    Substitution, TraitEnvironment, Ty, TyBuilder, TyExt,
};

use super::{
    const_as_usize, return_slot, AggregateKind, BinOp, CastKind, LocalId, MirBody, MirLowerError,
    Operand, Place, ProjectionElem, Rvalue, StatementKind, Terminator, UnOp,
};

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(x) => x,
            Err(_) => return Err(MirEvalError::TypeError("mismatched size")),
        }))
    };
}

#[derive(Debug, Default)]
struct VTableMap {
    ty_to_id: HashMap<Ty, usize>,
    id_to_ty: Vec<Ty>,
}

impl VTableMap {
    fn id(&mut self, ty: Ty) -> usize {
        if let Some(x) = self.ty_to_id.get(&ty) {
            return *x;
        }
        let id = self.id_to_ty.len();
        self.id_to_ty.push(ty.clone());
        self.ty_to_id.insert(ty, id);
        id
    }

    fn ty(&self, id: usize) -> Result<&Ty> {
        self.id_to_ty.get(id).ok_or(MirEvalError::InvalidVTableId(id))
    }

    fn ty_of_bytes(&self, bytes: &[u8]) -> Result<&Ty> {
        let id = from_bytes!(usize, bytes);
        self.ty(id)
    }
}

pub struct Evaluator<'a> {
    db: &'a dyn HirDatabase,
    trait_env: Arc<TraitEnvironment>,
    stack: Vec<u8>,
    heap: Vec<u8>,
    /// We don't really have function pointers, i.e. pointers to some assembly instructions that we can run. Instead, we
    /// store the type as an interned id in place of function and vtable pointers, and we recover back the type at the
    /// time of use.
    vtable_map: VTableMap,
    crate_id: CrateId,
    // FIXME: This is a workaround, see the comment on `interpret_mir`
    assert_placeholder_ty_is_unused: bool,
    /// A general limit on execution, to prevent non terminating programs from breaking r-a main process
    execution_limit: usize,
    /// An additional limit on stack depth, to prevent stack overflow
    stack_depth_limit: usize,
}

#[derive(Debug, Clone, Copy)]
enum Address {
    Stack(usize),
    Heap(usize),
    Invalid(usize),
}

use Address::*;

#[derive(Debug, Clone, Copy)]
struct Interval {
    addr: Address,
    size: usize,
}

#[derive(Debug, Clone)]
struct IntervalAndTy {
    interval: Interval,
    ty: Ty,
}

impl Interval {
    fn new(addr: Address, size: usize) -> Self {
        Self { addr, size }
    }

    fn get<'a>(&self, memory: &'a Evaluator<'a>) -> Result<&'a [u8]> {
        memory.read_memory(self.addr, self.size)
    }

    fn write_from_bytes(&self, memory: &mut Evaluator<'_>, bytes: &[u8]) -> Result<()> {
        memory.write_memory(self.addr, bytes)
    }

    fn write_from_interval(&self, memory: &mut Evaluator<'_>, interval: Interval) -> Result<()> {
        // FIXME: this could be more efficient
        let bytes = &interval.get(memory)?.to_vec();
        memory.write_memory(self.addr, bytes)
    }

    fn slice(self, range: Range<usize>) -> Interval {
        Interval { addr: self.addr.offset(range.start), size: range.len() }
    }
}

impl IntervalAndTy {
    fn get<'a>(&self, memory: &'a Evaluator<'a>) -> Result<&'a [u8]> {
        memory.read_memory(self.interval.addr, self.interval.size)
    }

    fn new(
        addr: Address,
        ty: Ty,
        evaluator: &Evaluator<'_>,
        locals: &Locals<'_>,
    ) -> Result<IntervalAndTy> {
        let size = evaluator.size_of_sized(&ty, locals, "type of interval")?;
        Ok(IntervalAndTy { interval: Interval { addr, size }, ty })
    }
}

enum IntervalOrOwned {
    Owned(Vec<u8>),
    Borrowed(Interval),
}
impl IntervalOrOwned {
    pub(crate) fn to_vec(self, memory: &Evaluator<'_>) -> Result<Vec<u8>> {
        Ok(match self {
            IntervalOrOwned::Owned(o) => o,
            IntervalOrOwned::Borrowed(b) => b.get(memory)?.to_vec(),
        })
    }
}

impl Address {
    fn from_bytes(x: &[u8]) -> Result<Self> {
        Ok(Address::from_usize(from_bytes!(usize, x)))
    }

    fn from_usize(x: usize) -> Self {
        if x > usize::MAX / 2 {
            Stack(x - usize::MAX / 2)
        } else if x > usize::MAX / 4 {
            Heap(x - usize::MAX / 4)
        } else {
            Invalid(x)
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        usize::to_le_bytes(self.to_usize()).to_vec()
    }

    fn to_usize(&self) -> usize {
        let as_num = match self {
            Stack(x) => *x + usize::MAX / 2,
            Heap(x) => *x + usize::MAX / 4,
            Invalid(x) => *x,
        };
        as_num
    }

    fn map(&self, f: impl FnOnce(usize) -> usize) -> Address {
        match self {
            Stack(x) => Stack(f(*x)),
            Heap(x) => Heap(f(*x)),
            Invalid(x) => Invalid(f(*x)),
        }
    }

    fn offset(&self, offset: usize) -> Address {
        self.map(|x| x + offset)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum MirEvalError {
    ConstEvalError(Box<ConstEvalError>),
    LayoutError(LayoutError, Ty),
    /// Means that code had type errors (or mismatched args) and we shouldn't generate mir in first place.
    TypeError(&'static str),
    /// Means that code had undefined behavior. We don't try to actively detect UB, but if it was detected
    /// then use this type of error.
    UndefinedBehavior(&'static str),
    Panic(String),
    MirLowerError(FunctionId, MirLowerError),
    MirLowerErrorForClosure(ClosureId, MirLowerError),
    TypeIsUnsized(Ty, &'static str),
    NotSupported(String),
    InvalidConst(Const),
    InFunction(FunctionId, Box<MirEvalError>),
    ExecutionLimitExceeded,
    StackOverflow,
    TargetDataLayoutNotAvailable,
    InvalidVTableId(usize),
}

impl std::fmt::Debug for MirEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstEvalError(arg0) => f.debug_tuple("ConstEvalError").field(arg0).finish(),
            Self::LayoutError(arg0, arg1) => {
                f.debug_tuple("LayoutError").field(arg0).field(arg1).finish()
            }
            Self::TypeError(arg0) => f.debug_tuple("TypeError").field(arg0).finish(),
            Self::UndefinedBehavior(arg0) => {
                f.debug_tuple("UndefinedBehavior").field(arg0).finish()
            }
            Self::Panic(msg) => write!(f, "Panic with message:\n{msg:?}"),
            Self::TargetDataLayoutNotAvailable => write!(f, "TargetDataLayoutNotAvailable"),
            Self::TypeIsUnsized(ty, it) => write!(f, "{ty:?} is unsized. {it} should be sized."),
            Self::ExecutionLimitExceeded => write!(f, "execution limit exceeded"),
            Self::StackOverflow => write!(f, "stack overflow"),
            Self::MirLowerError(arg0, arg1) => {
                f.debug_tuple("MirLowerError").field(arg0).field(arg1).finish()
            }
            Self::MirLowerErrorForClosure(arg0, arg1) => {
                f.debug_tuple("MirLowerError").field(arg0).field(arg1).finish()
            }
            Self::InvalidVTableId(arg0) => f.debug_tuple("InvalidVTableId").field(arg0).finish(),
            Self::NotSupported(arg0) => f.debug_tuple("NotSupported").field(arg0).finish(),
            Self::InvalidConst(arg0) => {
                let data = &arg0.data(Interner);
                f.debug_struct("InvalidConst").field("ty", &data.ty).field("value", &arg0).finish()
            }
            Self::InFunction(func, e) => {
                let mut e = &**e;
                let mut stack = vec![*func];
                while let Self::InFunction(f, next_e) = e {
                    e = &next_e;
                    stack.push(*f);
                }
                f.debug_struct("WithStack").field("error", e).field("stack", &stack).finish()
            }
        }
    }
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirEvalError::NotSupported(format!($x)))
    };
}

impl From<ConstEvalError> for MirEvalError {
    fn from(value: ConstEvalError) -> Self {
        match value {
            _ => MirEvalError::ConstEvalError(Box::new(value)),
        }
    }
}

type Result<T> = std::result::Result<T, MirEvalError>;

struct Locals<'a> {
    ptr: &'a ArenaMap<LocalId, Address>,
    body: &'a MirBody,
    subst: &'a Substitution,
}

pub fn interpret_mir(
    db: &dyn HirDatabase,
    body: &MirBody,
    subst: Substitution,
    // FIXME: This is workaround. Ideally, const generics should have a separate body (issue #7434), but now
    // they share their body with their parent, so in MIR lowering we have locals of the parent body, which
    // might have placeholders. With this argument, we (wrongly) assume that every placeholder type has
    // a zero size, hoping that they are all outside of our current body. Even without a fix for #7434, we can
    // (and probably should) do better here, for example by excluding bindings outside of the target expression.
    assert_placeholder_ty_is_unused: bool,
) -> Result<Const> {
    let ty = body.locals[return_slot()].ty.clone();
    let mut evaluator = Evaluator::new(db, body, assert_placeholder_ty_is_unused);
    let bytes = evaluator.interpret_mir(&body, None.into_iter(), subst.clone())?;
    let memory_map = evaluator.create_memory_map(
        &bytes,
        &ty,
        &Locals { ptr: &ArenaMap::new(), body: &body, subst: &subst },
    )?;
    return Ok(intern_const_scalar(ConstScalar::Bytes(bytes, memory_map), ty));
}

impl Evaluator<'_> {
    pub fn new<'a>(
        db: &'a dyn HirDatabase,
        body: &MirBody,
        assert_placeholder_ty_is_unused: bool,
    ) -> Evaluator<'a> {
        let crate_id = body.owner.module(db.upcast()).krate();
        let trait_env = db.trait_environment_for_body(body.owner);
        Evaluator {
            stack: vec![0],
            heap: vec![0],
            vtable_map: VTableMap::default(),
            db,
            trait_env,
            crate_id,
            assert_placeholder_ty_is_unused,
            stack_depth_limit: 100,
            execution_limit: 100_000,
        }
    }

    fn place_addr(&self, p: &Place, locals: &Locals<'_>) -> Result<Address> {
        Ok(self.place_addr_and_ty_and_metadata(p, locals)?.0)
    }

    fn place_interval(&self, p: &Place, locals: &Locals<'_>) -> Result<Interval> {
        let place_addr_and_ty = self.place_addr_and_ty_and_metadata(p, locals)?;
        Ok(Interval {
            addr: place_addr_and_ty.0,
            size: self.size_of_sized(
                &place_addr_and_ty.1,
                locals,
                "Type of place that we need its interval",
            )?,
        })
    }

    fn ptr_size(&self) -> usize {
        match self.db.target_data_layout(self.crate_id) {
            Some(x) => x.pointer_size.bytes_usize(),
            None => 8,
        }
    }

    fn place_addr_and_ty_and_metadata<'a>(
        &'a self,
        p: &Place,
        locals: &'a Locals<'a>,
    ) -> Result<(Address, Ty, Option<Interval>)> {
        let mut addr = locals.ptr[p.local];
        let mut ty: Ty =
            self.ty_filler(&locals.body.locals[p.local].ty, locals.subst, locals.body.owner)?;
        let mut metadata = None; // locals are always sized
        for proj in &p.projection {
            let prev_ty = ty.clone();
            ty = proj.projected_ty(ty, self.db, |c, f| {
                let (def, _) = self.db.lookup_intern_closure(c.into());
                let infer = self.db.infer(def);
                let (captures, _) = infer.closure_info(&c);
                captures.get(f).expect("broken closure field").ty.clone()
            });
            match proj {
                ProjectionElem::Deref => {
                    metadata = if self.size_of(&ty, locals)?.is_none() {
                        Some(Interval { addr: addr.offset(self.ptr_size()), size: self.ptr_size() })
                    } else {
                        None
                    };
                    let x = from_bytes!(usize, self.read_memory(addr, self.ptr_size())?);
                    addr = Address::from_usize(x);
                }
                ProjectionElem::Index(op) => {
                    let offset =
                        from_bytes!(usize, self.read_memory(locals.ptr[*op], self.ptr_size())?);
                    metadata = None; // Result of index is always sized
                    let ty_size =
                        self.size_of_sized(&ty, locals, "array inner type should be sized")?;
                    addr = addr.offset(ty_size * offset);
                }
                &ProjectionElem::TupleOrClosureField(f) => {
                    let layout = self.layout(&prev_ty)?;
                    let offset = layout.fields.offset(f).bytes_usize();
                    addr = addr.offset(offset);
                    metadata = None; // tuple field is always sized
                }
                ProjectionElem::Field(f) => {
                    let layout = self.layout(&prev_ty)?;
                    let variant_layout = match &layout.variants {
                        Variants::Single { .. } => &layout,
                        Variants::Multiple { variants, .. } => {
                            &variants[match f.parent {
                                hir_def::VariantId::EnumVariantId(x) => {
                                    RustcEnumVariantIdx(x.local_id)
                                }
                                _ => {
                                    return Err(MirEvalError::TypeError(
                                        "Multivariant layout only happens for enums",
                                    ))
                                }
                            }]
                        }
                    };
                    let offset = variant_layout
                        .fields
                        .offset(u32::from(f.local_id.into_raw()) as usize)
                        .bytes_usize();
                    addr = addr.offset(offset);
                    // FIXME: support structs with unsized fields
                    metadata = None;
                }
                ProjectionElem::ConstantIndex { .. } => {
                    not_supported!("constant index")
                }
                ProjectionElem::Subslice { .. } => not_supported!("subslice"),
                ProjectionElem::OpaqueCast(_) => not_supported!("opaque cast"),
            }
        }
        Ok((addr, ty, metadata))
    }

    fn layout(&self, ty: &Ty) -> Result<Layout> {
        layout_of_ty(self.db, ty, self.crate_id)
            .map_err(|e| MirEvalError::LayoutError(e, ty.clone()))
    }

    fn layout_adt(&self, adt: AdtId, subst: Substitution) -> Result<Layout> {
        self.db.layout_of_adt(adt, subst.clone()).map_err(|e| {
            MirEvalError::LayoutError(e, TyKind::Adt(chalk_ir::AdtId(adt), subst).intern(Interner))
        })
    }

    fn place_ty<'a>(&'a self, p: &Place, locals: &'a Locals<'a>) -> Result<Ty> {
        Ok(self.place_addr_and_ty_and_metadata(p, locals)?.1)
    }

    fn operand_ty(&self, o: &Operand, locals: &Locals<'_>) -> Result<Ty> {
        Ok(match o {
            Operand::Copy(p) | Operand::Move(p) => self.place_ty(p, locals)?,
            Operand::Constant(c) => c.data(Interner).ty.clone(),
        })
    }

    fn operand_ty_and_eval(&mut self, o: &Operand, locals: &Locals<'_>) -> Result<IntervalAndTy> {
        Ok(IntervalAndTy {
            interval: self.eval_operand(o, locals)?,
            ty: self.operand_ty(o, locals)?,
        })
    }

    fn interpret_mir(
        &mut self,
        body: &MirBody,
        args: impl Iterator<Item = Vec<u8>>,
        subst: Substitution,
    ) -> Result<Vec<u8>> {
        if let Some(x) = self.stack_depth_limit.checked_sub(1) {
            self.stack_depth_limit = x;
        } else {
            return Err(MirEvalError::StackOverflow);
        }
        let mut current_block_idx = body.start_block;
        let mut locals = Locals { ptr: &ArenaMap::new(), body: &body, subst: &subst };
        let (locals_ptr, stack_size) = {
            let mut stack_ptr = self.stack.len();
            let addr = body
                .locals
                .iter()
                .map(|(id, x)| {
                    let size =
                        self.size_of_sized(&x.ty, &locals, "no unsized local in extending stack")?;
                    let my_ptr = stack_ptr;
                    stack_ptr += size;
                    Ok((id, Stack(my_ptr)))
                })
                .collect::<Result<ArenaMap<LocalId, _>>>()?;
            let stack_size = stack_ptr - self.stack.len();
            (addr, stack_size)
        };
        locals.ptr = &locals_ptr;
        self.stack.extend(iter::repeat(0).take(stack_size));
        let mut remain_args = body.arg_count;
        for ((_, addr), value) in locals_ptr.iter().skip(1).zip(args) {
            self.write_memory(*addr, &value)?;
            if remain_args == 0 {
                return Err(MirEvalError::TypeError("more arguments provided"));
            }
            remain_args -= 1;
        }
        if remain_args > 0 {
            return Err(MirEvalError::TypeError("not enough arguments provided"));
        }
        loop {
            let current_block = &body.basic_blocks[current_block_idx];
            if let Some(x) = self.execution_limit.checked_sub(1) {
                self.execution_limit = x;
            } else {
                return Err(MirEvalError::ExecutionLimitExceeded);
            }
            for statement in &current_block.statements {
                match &statement.kind {
                    StatementKind::Assign(l, r) => {
                        let addr = self.place_addr(l, &locals)?;
                        let result = self.eval_rvalue(r, &locals)?.to_vec(&self)?;
                        self.write_memory(addr, &result)?;
                    }
                    StatementKind::Deinit(_) => not_supported!("de-init statement"),
                    StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::Nop => (),
                }
            }
            let Some(terminator) = current_block.terminator.as_ref() else {
                not_supported!("block without terminator");
            };
            match terminator {
                Terminator::Goto { target } => {
                    current_block_idx = *target;
                }
                Terminator::Call {
                    func,
                    args,
                    destination,
                    target,
                    cleanup: _,
                    from_hir_call: _,
                } => {
                    let destination = self.place_interval(destination, &locals)?;
                    let fn_ty = self.operand_ty(func, &locals)?;
                    let args = args
                        .iter()
                        .map(|x| self.operand_ty_and_eval(x, &locals))
                        .collect::<Result<Vec<_>>>()?;
                    match &fn_ty.data(Interner).kind {
                        TyKind::Function(_) => {
                            let bytes = self.eval_operand(func, &locals)?;
                            self.exec_fn_pointer(bytes, destination, &args, &locals)?;
                        }
                        TyKind::FnDef(def, generic_args) => {
                            self.exec_fn_def(*def, generic_args, destination, &args, &locals)?;
                        }
                        x => not_supported!("unknown function type {x:?}"),
                    }
                    current_block_idx = target.expect("broken mir, function without target");
                }
                Terminator::SwitchInt { discr, targets } => {
                    let val = u128::from_le_bytes(pad16(
                        self.eval_operand(discr, &locals)?.get(&self)?,
                        false,
                    ));
                    current_block_idx = targets.target_for_value(val);
                }
                Terminator::Return => {
                    let ty = body.locals[return_slot()].ty.clone();
                    self.stack_depth_limit += 1;
                    return Ok(self
                        .read_memory(
                            locals.ptr[return_slot()],
                            self.size_of_sized(&ty, &locals, "return type")?,
                        )?
                        .to_owned());
                }
                Terminator::Unreachable => {
                    return Err(MirEvalError::UndefinedBehavior("unreachable executed"));
                }
                _ => not_supported!("unknown terminator"),
            }
        }
    }

    fn eval_rvalue<'a>(
        &'a mut self,
        r: &'a Rvalue,
        locals: &'a Locals<'a>,
    ) -> Result<IntervalOrOwned> {
        use IntervalOrOwned::*;
        Ok(match r {
            Rvalue::Use(x) => Borrowed(self.eval_operand(x, locals)?),
            Rvalue::Ref(_, p) => {
                let (addr, _, metadata) = self.place_addr_and_ty_and_metadata(p, locals)?;
                let mut r = addr.to_bytes();
                if let Some(metadata) = metadata {
                    r.extend(metadata.get(self)?);
                }
                Owned(r)
            }
            Rvalue::Len(_) => not_supported!("rvalue len"),
            Rvalue::UnaryOp(op, val) => {
                let mut c = self.eval_operand(val, locals)?.get(&self)?;
                let mut ty = self.operand_ty(val, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = self.size_of_sized(&ty, locals, "operand of unary op")?;
                    c = self.read_memory(Address::from_bytes(c)?, size)?;
                }
                let mut c = c.to_vec();
                if ty.as_builtin() == Some(BuiltinType::Bool) {
                    c[0] = 1 - c[0];
                } else {
                    match op {
                        UnOp::Not => c.iter_mut().for_each(|x| *x = !*x),
                        UnOp::Neg => {
                            c.iter_mut().for_each(|x| *x = !*x);
                            for k in c.iter_mut() {
                                let o;
                                (*k, o) = k.overflowing_add(1);
                                if !o {
                                    break;
                                }
                            }
                        }
                    }
                }
                Owned(c)
            }
            Rvalue::CheckedBinaryOp(op, lhs, rhs) => {
                let lc = self.eval_operand(lhs, locals)?;
                let rc = self.eval_operand(rhs, locals)?;
                let mut lc = lc.get(&self)?;
                let mut rc = rc.get(&self)?;
                let mut ty = self.operand_ty(lhs, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = if ty.kind(Interner) == &TyKind::Str {
                        let ns = from_bytes!(usize, &lc[self.ptr_size()..self.ptr_size() * 2]);
                        lc = &lc[..self.ptr_size()];
                        rc = &rc[..self.ptr_size()];
                        ns
                    } else {
                        self.size_of_sized(&ty, locals, "operand of binary op")?
                    };
                    lc = self.read_memory(Address::from_bytes(lc)?, size)?;
                    rc = self.read_memory(Address::from_bytes(rc)?, size)?;
                }
                let is_signed = matches!(ty.as_builtin(), Some(BuiltinType::Int(_)));
                let l128 = i128::from_le_bytes(pad16(lc, is_signed));
                let r128 = i128::from_le_bytes(pad16(rc, is_signed));
                match op {
                    BinOp::Ge | BinOp::Gt | BinOp::Le | BinOp::Lt | BinOp::Eq | BinOp::Ne => {
                        let r = match op {
                            BinOp::Ge => l128 >= r128,
                            BinOp::Gt => l128 > r128,
                            BinOp::Le => l128 <= r128,
                            BinOp::Lt => l128 < r128,
                            BinOp::Eq => l128 == r128,
                            BinOp::Ne => l128 != r128,
                            _ => unreachable!(),
                        };
                        let r = r as u8;
                        Owned(vec![r])
                    }
                    BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::Add
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Rem
                    | BinOp::Sub => {
                        let r = match op {
                            BinOp::Add => l128.overflowing_add(r128).0,
                            BinOp::Mul => l128.overflowing_mul(r128).0,
                            BinOp::Div => l128.checked_div(r128).ok_or_else(|| {
                                MirEvalError::Panic(format!("Overflow in {op:?}"))
                            })?,
                            BinOp::Rem => l128.checked_rem(r128).ok_or_else(|| {
                                MirEvalError::Panic(format!("Overflow in {op:?}"))
                            })?,
                            BinOp::Sub => l128.overflowing_sub(r128).0,
                            BinOp::BitAnd => l128 & r128,
                            BinOp::BitOr => l128 | r128,
                            BinOp::BitXor => l128 ^ r128,
                            _ => unreachable!(),
                        };
                        let r = r.to_le_bytes();
                        for &k in &r[lc.len()..] {
                            if k != 0 && (k != 255 || !is_signed) {
                                return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                            }
                        }
                        Owned(r[0..lc.len()].into())
                    }
                    BinOp::Shl | BinOp::Shr => {
                        let shift_amount = if r128 < 0 {
                            return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                        } else if r128 > 128 {
                            return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                        } else {
                            r128 as u8
                        };
                        let r = match op {
                            BinOp::Shl => l128 << shift_amount,
                            BinOp::Shr => l128 >> shift_amount,
                            _ => unreachable!(),
                        };
                        Owned(r.to_le_bytes()[0..lc.len()].into())
                    }
                    BinOp::Offset => not_supported!("offset binop"),
                }
            }
            Rvalue::Discriminant(p) => {
                let ty = self.place_ty(p, locals)?;
                let bytes = self.eval_place(p, locals)?.get(&self)?;
                let layout = self.layout(&ty)?;
                let enum_id = 'b: {
                    match ty.kind(Interner) {
                        TyKind::Adt(e, _) => match e.0 {
                            AdtId::EnumId(e) => break 'b e,
                            _ => (),
                        },
                        _ => (),
                    }
                    return Ok(Owned(0u128.to_le_bytes().to_vec()));
                };
                match layout.variants {
                    Variants::Single { index } => {
                        let r = self.db.const_eval_discriminant(EnumVariantId {
                            parent: enum_id,
                            local_id: index.0,
                        })?;
                        Owned(r.to_le_bytes().to_vec())
                    }
                    Variants::Multiple { tag, tag_encoding, .. } => {
                        let Some(target_data_layout) = self.db.target_data_layout(self.crate_id) else {
                            not_supported!("missing target data layout");
                        };
                        let size = tag.size(&*target_data_layout).bytes_usize();
                        let offset = layout.fields.offset(0).bytes_usize(); // The only field on enum variants is the tag field
                        match tag_encoding {
                            TagEncoding::Direct => {
                                let tag = &bytes[offset..offset + size];
                                Owned(pad16(tag, false).to_vec())
                            }
                            TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                                let tag = &bytes[offset..offset + size];
                                let candidate_discriminant = i128::from_le_bytes(pad16(tag, false))
                                    .wrapping_sub(niche_start as i128);
                                let enum_data = self.db.enum_data(enum_id);
                                let result = 'b: {
                                    for (local_id, _) in enum_data.variants.iter() {
                                        if candidate_discriminant
                                            == self.db.const_eval_discriminant(EnumVariantId {
                                                parent: enum_id,
                                                local_id,
                                            })?
                                        {
                                            break 'b candidate_discriminant;
                                        }
                                    }
                                    self.db.const_eval_discriminant(EnumVariantId {
                                        parent: enum_id,
                                        local_id: untagged_variant.0,
                                    })?
                                };
                                Owned(result.to_le_bytes().to_vec())
                            }
                        }
                    }
                }
            }
            Rvalue::ShallowInitBox(_, _) => not_supported!("shallow init box"),
            Rvalue::CopyForDeref(_) => not_supported!("copy for deref"),
            Rvalue::Aggregate(kind, values) => {
                let values = values
                    .iter()
                    .map(|x| self.eval_operand(x, locals))
                    .collect::<Result<Vec<_>>>()?;
                match kind {
                    AggregateKind::Array(_) => {
                        let mut r = vec![];
                        for x in values {
                            let value = x.get(&self)?;
                            r.extend(value);
                        }
                        Owned(r)
                    }
                    AggregateKind::Tuple(ty) => {
                        let layout = self.layout(&ty)?;
                        Owned(self.make_by_layout(
                            layout.size.bytes_usize(),
                            &layout,
                            None,
                            values.iter().copied(),
                        )?)
                    }
                    AggregateKind::Union(x, f) => {
                        let layout = self.layout_adt((*x).into(), Substitution::empty(Interner))?;
                        let offset = layout
                            .fields
                            .offset(u32::from(f.local_id.into_raw()) as usize)
                            .bytes_usize();
                        let op = values[0].get(&self)?;
                        let mut result = vec![0; layout.size.bytes_usize()];
                        result[offset..offset + op.len()].copy_from_slice(op);
                        Owned(result)
                    }
                    AggregateKind::Adt(x, subst) => {
                        let subst = self.subst_filler(subst, locals);
                        let (size, variant_layout, tag) =
                            self.layout_of_variant(*x, subst, locals)?;
                        Owned(self.make_by_layout(
                            size,
                            &variant_layout,
                            tag,
                            values.iter().copied(),
                        )?)
                    }
                    AggregateKind::Closure(ty) => {
                        let layout = self.layout(&ty)?;
                        Owned(self.make_by_layout(
                            layout.size.bytes_usize(),
                            &layout,
                            None,
                            values.iter().copied(),
                        )?)
                    }
                }
            }
            Rvalue::Cast(kind, operand, target_ty) => match kind {
                CastKind::Pointer(cast) => match cast {
                    PointerCast::ReifyFnPointer => {
                        let current_ty = self.operand_ty(operand, locals)?;
                        if let TyKind::FnDef(_, _) = &current_ty.data(Interner).kind {
                            let id = self.vtable_map.id(current_ty);
                            let ptr_size = self.ptr_size();
                            Owned(id.to_le_bytes()[0..ptr_size].to_vec())
                        } else {
                            not_supported!("ReifyFnPointer cast of a non FnDef type");
                        }
                    }
                    PointerCast::Unsize => {
                        let current_ty = self.operand_ty(operand, locals)?;
                        match &target_ty.data(Interner).kind {
                            TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => {
                                match &ty.data(Interner).kind {
                                    TyKind::Slice(_) => match &current_ty.data(Interner).kind {
                                        TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => {
                                            match &ty.data(Interner).kind {
                                                TyKind::Array(_, size) => {
                                                    let addr = self
                                                        .eval_operand(operand, locals)?
                                                        .get(&self)?;
                                                    let len = const_as_usize(size);
                                                    let mut r = Vec::with_capacity(16);
                                                    r.extend(addr.iter().copied());
                                                    r.extend(len.to_le_bytes().into_iter());
                                                    Owned(r)
                                                }
                                                _ => {
                                                    not_supported!("slice unsizing from non arrays")
                                                }
                                            }
                                        }
                                        _ => not_supported!("slice unsizing from non pointers"),
                                    },
                                    TyKind::Dyn(_) => match &current_ty.data(Interner).kind {
                                        TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => {
                                            let vtable = self.vtable_map.id(ty.clone());
                                            let addr =
                                                self.eval_operand(operand, locals)?.get(&self)?;
                                            let mut r = Vec::with_capacity(16);
                                            r.extend(addr.iter().copied());
                                            r.extend(vtable.to_le_bytes().into_iter());
                                            Owned(r)
                                        }
                                        _ => not_supported!("dyn unsizing from non pointers"),
                                    },
                                    _ => not_supported!("unknown unsized cast"),
                                }
                            }
                            _ => not_supported!("unsized cast on unknown pointer type"),
                        }
                    }
                    x => not_supported!("pointer cast {x:?}"),
                },
                CastKind::DynStar => not_supported!("dyn star cast"),
                CastKind::IntToInt
                | CastKind::PointerExposeAddress
                | CastKind::PointerFromExposedAddress => {
                    // FIXME: handle signed cast
                    let current = pad16(self.eval_operand(operand, locals)?.get(&self)?, false);
                    let dest_size =
                        self.size_of_sized(target_ty, locals, "destination of int to int cast")?;
                    Owned(current[0..dest_size].to_vec())
                }
                CastKind::FloatToInt => not_supported!("float to int cast"),
                CastKind::FloatToFloat => not_supported!("float to float cast"),
                CastKind::IntToFloat => not_supported!("float to int cast"),
                CastKind::PtrToPtr => {
                    let current = pad16(self.eval_operand(operand, locals)?.get(&self)?, false);
                    let dest_size =
                        self.size_of_sized(target_ty, locals, "destination of ptr to ptr cast")?;
                    Owned(current[0..dest_size].to_vec())
                }
                CastKind::FnPtrToPtr => not_supported!("fn ptr to ptr cast"),
            },
        })
    }

    fn layout_of_variant(
        &mut self,
        x: VariantId,
        subst: Substitution,
        locals: &Locals<'_>,
    ) -> Result<(usize, Layout, Option<(usize, usize, i128)>)> {
        let adt = x.adt_id();
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let VariantId::EnumVariantId(x) = x {
                if AdtId::from(f.parent) == adt {
                    // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                    // infinite sized type errors) we use a dummy layout
                    let i = self.db.const_eval_discriminant(x)?;
                    return Ok((16, self.layout(&TyBuilder::unit())?, Some((0, 16, i))));
                }
            }
        }
        let layout = self.layout_adt(adt, subst)?;
        Ok(match layout.variants {
            Variants::Single { .. } => (layout.size.bytes_usize(), layout, None),
            Variants::Multiple { variants, tag, tag_encoding, .. } => {
                let cx = self
                    .db
                    .target_data_layout(self.crate_id)
                    .ok_or(MirEvalError::TargetDataLayoutNotAvailable)?;
                let enum_variant_id = match x {
                    VariantId::EnumVariantId(x) => x,
                    _ => not_supported!("multi variant layout for non-enums"),
                };
                let rustc_enum_variant_idx = RustcEnumVariantIdx(enum_variant_id.local_id);
                let mut discriminant = self.db.const_eval_discriminant(enum_variant_id)?;
                let variant_layout = variants[rustc_enum_variant_idx].clone();
                let have_tag = match tag_encoding {
                    TagEncoding::Direct => true,
                    TagEncoding::Niche { untagged_variant, niche_variants: _, niche_start } => {
                        discriminant = discriminant.wrapping_add(niche_start as i128);
                        untagged_variant != rustc_enum_variant_idx
                    }
                };
                (
                    layout.size.bytes_usize(),
                    variant_layout,
                    if have_tag {
                        Some((
                            layout.fields.offset(0).bytes_usize(),
                            tag.size(&*cx).bytes_usize(),
                            discriminant,
                        ))
                    } else {
                        None
                    },
                )
            }
        })
    }

    fn make_by_layout(
        &mut self,
        size: usize, // Not necessarily equal to variant_layout.size
        variant_layout: &Layout,
        tag: Option<(usize, usize, i128)>,
        values: impl Iterator<Item = Interval>,
    ) -> Result<Vec<u8>> {
        let mut result = vec![0; size];
        if let Some((offset, size, value)) = tag {
            result[offset..offset + size].copy_from_slice(&value.to_le_bytes()[0..size]);
        }
        for (i, op) in values.enumerate() {
            let offset = variant_layout.fields.offset(i).bytes_usize();
            let op = op.get(&self)?;
            result[offset..offset + op.len()].copy_from_slice(op);
        }
        Ok(result)
    }

    fn eval_operand(&mut self, x: &Operand, locals: &Locals<'_>) -> Result<Interval> {
        Ok(match x {
            Operand::Copy(p) | Operand::Move(p) => self.eval_place(p, locals)?,
            Operand::Constant(konst) => {
                let data = &konst.data(Interner);
                match &data.value {
                    chalk_ir::ConstValue::BoundVar(b) => {
                        let c = locals
                            .subst
                            .as_slice(Interner)
                            .get(b.index)
                            .ok_or(MirEvalError::TypeError("missing generic arg"))?
                            .assert_const_ref(Interner);
                        self.eval_operand(&Operand::Constant(c.clone()), locals)?
                    }
                    chalk_ir::ConstValue::InferenceVar(_) => {
                        not_supported!("inference var constant")
                    }
                    chalk_ir::ConstValue::Placeholder(_) => not_supported!("placeholder constant"),
                    chalk_ir::ConstValue::Concrete(c) => match &c.interned {
                        ConstScalar::Bytes(v, memory_map) => {
                            let mut v: Cow<'_, [u8]> = Cow::Borrowed(v);
                            let patch_map = memory_map.transform_addresses(|b| {
                                let addr = self.heap_allocate(b.len());
                                self.write_memory(addr, b)?;
                                Ok(addr.to_usize())
                            })?;
                            let size = self.size_of(&data.ty, locals)?.unwrap_or(v.len());
                            if size != v.len() {
                                // Handle self enum
                                if size == 16 && v.len() < 16 {
                                    v = Cow::Owned(pad16(&v, false).to_vec());
                                } else if size < 16 && v.len() == 16 {
                                    v = Cow::Owned(v[0..size].to_vec());
                                } else {
                                    return Err(MirEvalError::InvalidConst(konst.clone()));
                                }
                            }
                            let addr = self.heap_allocate(size);
                            self.write_memory(addr, &v)?;
                            self.patch_addresses(&patch_map, addr, &data.ty, locals)?;
                            Interval::new(addr, size)
                        }
                        ConstScalar::Unknown => not_supported!("evaluating unknown const"),
                    },
                }
            }
        })
    }

    fn eval_place(&mut self, p: &Place, locals: &Locals<'_>) -> Result<Interval> {
        let addr = self.place_addr(p, locals)?;
        Ok(Interval::new(
            addr,
            self.size_of_sized(&self.place_ty(p, locals)?, locals, "type of this place")?,
        ))
    }

    fn read_memory(&self, addr: Address, size: usize) -> Result<&[u8]> {
        let (mem, pos) = match addr {
            Stack(x) => (&self.stack, x),
            Heap(x) => (&self.heap, x),
            Invalid(_) => {
                return Err(MirEvalError::UndefinedBehavior("read invalid memory address"))
            }
        };
        mem.get(pos..pos + size).ok_or(MirEvalError::UndefinedBehavior("out of bound memory read"))
    }

    fn write_memory(&mut self, addr: Address, r: &[u8]) -> Result<()> {
        let (mem, pos) = match addr {
            Stack(x) => (&mut self.stack, x),
            Heap(x) => (&mut self.heap, x),
            Invalid(_) => {
                return Err(MirEvalError::UndefinedBehavior("write invalid memory address"))
            }
        };
        mem.get_mut(pos..pos + r.len())
            .ok_or(MirEvalError::UndefinedBehavior("out of bound memory write"))?
            .copy_from_slice(r);
        Ok(())
    }

    fn size_of(&self, ty: &Ty, locals: &Locals<'_>) -> Result<Option<usize>> {
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let Some((adt, _)) = ty.as_adt() {
                if AdtId::from(f.parent) == adt {
                    // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                    // infinite sized type errors) we use a dummy size
                    return Ok(Some(16));
                }
            }
        }
        let ty = &self.ty_filler(ty, locals.subst, locals.body.owner)?;
        let layout = self.layout(ty);
        if self.assert_placeholder_ty_is_unused {
            if matches!(layout, Err(MirEvalError::LayoutError(LayoutError::HasPlaceholder, _))) {
                return Ok(Some(0));
            }
        }
        let layout = layout?;
        Ok(layout.is_sized().then(|| layout.size.bytes_usize()))
    }

    /// A version of `self.size_of` which returns error if the type is unsized. `what` argument should
    /// be something that complete this: `error: type {ty} was unsized. {what} should be sized`
    fn size_of_sized(&self, ty: &Ty, locals: &Locals<'_>, what: &'static str) -> Result<usize> {
        match self.size_of(ty, locals)? {
            Some(x) => Ok(x),
            None => Err(MirEvalError::TypeIsUnsized(ty.clone(), what)),
        }
    }

    /// Uses `ty_filler` to fill an entire subst
    fn subst_filler(&self, subst: &Substitution, locals: &Locals<'_>) -> Substitution {
        Substitution::from_iter(
            Interner,
            subst.iter(Interner).map(|x| match x.data(Interner) {
                chalk_ir::GenericArgData::Ty(ty) => {
                    let Ok(ty) = self.ty_filler(ty, locals.subst, locals.body.owner) else {
                        return x.clone();
                    };
                    chalk_ir::GenericArgData::Ty(ty).intern(Interner)
                }
                _ => x.clone(),
            }),
        )
    }

    /// This function substitutes placeholders of the body with the provided subst, effectively plays
    /// the rule of monomorphization. In addition to placeholders, it substitutes opaque types (return
    /// position impl traits) with their underlying type.
    fn ty_filler(&self, ty: &Ty, subst: &Substitution, owner: DefWithBodyId) -> Result<Ty> {
        struct Filler<'a> {
            db: &'a dyn HirDatabase,
            subst: &'a Substitution,
            skip_params: usize,
        }
        impl FallibleTypeFolder<Interner> for Filler<'_> {
            type Error = MirEvalError;

            fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
                self
            }

            fn interner(&self) -> Interner {
                Interner
            }

            fn try_fold_ty(
                &mut self,
                ty: Ty,
                outer_binder: DebruijnIndex,
            ) -> std::result::Result<Ty, Self::Error> {
                match ty.kind(Interner) {
                    TyKind::OpaqueType(id, subst) => {
                        let impl_trait_id = self.db.lookup_intern_impl_trait_id((*id).into());
                        match impl_trait_id {
                            crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                                let infer = self.db.infer(func.into());
                                let filler = &mut Filler { db: self.db, subst, skip_params: 0 };
                                filler.try_fold_ty(infer.type_of_rpit[idx].clone(), outer_binder)
                            }
                            crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                                not_supported!("async block impl trait");
                            }
                        }
                    }
                    _ => ty.try_super_fold_with(self.as_dyn(), outer_binder),
                }
            }

            fn try_fold_free_placeholder_ty(
                &mut self,
                idx: chalk_ir::PlaceholderIndex,
                _outer_binder: DebruijnIndex,
            ) -> std::result::Result<Ty, Self::Error> {
                let x = from_placeholder_idx(self.db, idx);
                Ok(self
                    .subst
                    .as_slice(Interner)
                    .get((u32::from(x.local_id.into_raw()) as usize) + self.skip_params)
                    .and_then(|x| x.ty(Interner))
                    .ok_or(MirEvalError::TypeError("Generic arg not provided"))?
                    .clone())
            }
        }
        let filler = &mut Filler { db: self.db, subst, skip_params: 0 };
        Ok(normalize(self.db, owner, ty.clone().try_fold_with(filler, DebruijnIndex::INNERMOST)?))
    }

    fn heap_allocate(&mut self, s: usize) -> Address {
        let pos = self.heap.len();
        self.heap.extend(iter::repeat(0).take(s));
        Address::Heap(pos)
    }

    pub fn interpret_mir_with_no_arg(&mut self, body: &MirBody) -> Result<Vec<u8>> {
        self.interpret_mir(&body, vec![].into_iter(), Substitution::empty(Interner))
    }

    fn detect_lang_function(&self, def: FunctionId) -> Option<LangItem> {
        use LangItem::*;
        let candidate = lang_attr(self.db.upcast(), def)?;
        // We want to execute these functions with special logic
        if [PanicFmt, BeginPanic, SliceLen].contains(&candidate) {
            return Some(candidate);
        }
        None
    }

    fn detect_fn_trait(&self, def: FunctionId) -> Option<FnTrait> {
        use LangItem::*;
        let ItemContainerId::TraitId(parent) = self.db.lookup_intern_function(def).container else {
            return None;
        };
        let l = lang_attr(self.db.upcast(), parent)?;
        match l {
            FnOnce => Some(FnTrait::FnOnce),
            FnMut => Some(FnTrait::FnMut),
            Fn => Some(FnTrait::Fn),
            _ => None,
        }
    }

    fn create_memory_map(&self, bytes: &[u8], ty: &Ty, locals: &Locals<'_>) -> Result<MemoryMap> {
        // FIXME: support indirect references
        let mut mm = MemoryMap::default();
        match ty.kind(Interner) {
            TyKind::Ref(_, _, t) => {
                let size = self.size_of(t, locals)?;
                match size {
                    Some(size) => {
                        let addr_usize = from_bytes!(usize, bytes);
                        mm.insert(
                            addr_usize,
                            self.read_memory(Address::from_usize(addr_usize), size)?.to_vec(),
                        )
                    }
                    None => {
                        let element_size = match t.kind(Interner) {
                            TyKind::Str => 1,
                            TyKind::Slice(t) => {
                                self.size_of_sized(t, locals, "slice inner type")?
                            }
                            _ => return Ok(mm), // FIXME: support other kind of unsized types
                        };
                        let (addr, meta) = bytes.split_at(bytes.len() / 2);
                        let size = element_size * from_bytes!(usize, meta);
                        let addr = Address::from_bytes(addr)?;
                        mm.insert(addr.to_usize(), self.read_memory(addr, size)?.to_vec());
                    }
                }
            }
            _ => (),
        }
        Ok(mm)
    }

    fn patch_addresses(
        &mut self,
        patch_map: &HashMap<usize, usize>,
        addr: Address,
        ty: &Ty,
        locals: &Locals<'_>,
    ) -> Result<()> {
        // FIXME: support indirect references
        let my_size = self.size_of_sized(ty, locals, "value to patch address")?;
        match ty.kind(Interner) {
            TyKind::Ref(_, _, t) => {
                let size = self.size_of(t, locals)?;
                match size {
                    Some(_) => {
                        let current = from_bytes!(usize, self.read_memory(addr, my_size)?);
                        if let Some(x) = patch_map.get(&current) {
                            self.write_memory(addr, &x.to_le_bytes())?;
                        }
                    }
                    None => {
                        let current = from_bytes!(usize, self.read_memory(addr, my_size / 2)?);
                        if let Some(x) = patch_map.get(&current) {
                            self.write_memory(addr, &x.to_le_bytes())?;
                        }
                    }
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn exec_intrinsic(
        &mut self,
        as_str: &str,
        args: &[IntervalAndTy],
        generic_args: Substitution,
        destination: Interval,
        locals: &Locals<'_>,
    ) -> Result<()> {
        match as_str {
            "size_of" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("size_of generic arg is not provided"));
                };
                let size = self.size_of_sized(ty, locals, "size_of arg")?;
                destination.write_from_bytes(self, &size.to_le_bytes()[0..destination.size])
            }
            "wrapping_add" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("const_eval_select args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_add(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "copy" | "copy_nonoverlapping" => {
                let [src, dst, offset] = args else {
                    return Err(MirEvalError::TypeError("copy_nonoverlapping args are not provided"));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("copy_nonoverlapping generic arg is not provided"));
                };
                let src = Address::from_bytes(src.get(self)?)?;
                let dst = Address::from_bytes(dst.get(self)?)?;
                let offset = from_bytes!(usize, offset.get(self)?);
                let size = self.size_of_sized(ty, locals, "copy_nonoverlapping ptr type")?;
                let size = offset * size;
                let src = Interval { addr: src, size };
                let dst = Interval { addr: dst, size };
                dst.write_from_interval(self, src)
            }
            "offset" | "arith_offset" => {
                let [ptr, offset] = args else {
                    return Err(MirEvalError::TypeError("offset args are not provided"));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("offset generic arg is not provided"));
                };
                let ptr = u128::from_le_bytes(pad16(ptr.get(self)?, false));
                let offset = u128::from_le_bytes(pad16(offset.get(self)?, false));
                let size = self.size_of_sized(ty, locals, "offset ptr type")? as u128;
                let ans = ptr + offset * size;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "assert_inhabited" | "assert_zero_valid" | "assert_uninit_valid" => {
                // FIXME: We should actually implement these checks
                Ok(())
            }
            "forget" => {
                // We don't call any drop glue yet, so there is nothing here
                Ok(())
            }
            "transmute" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("trasmute arg is not provided"));
                };
                destination.write_from_interval(self, arg.interval)
            }
            "const_eval_select" => {
                let [tuple, const_fn, _] = args else {
                    return Err(MirEvalError::TypeError("const_eval_select args are not provided"));
                };
                let mut args = vec![const_fn.clone()];
                let TyKind::Tuple(_, fields) = tuple.ty.kind(Interner) else {
                    return Err(MirEvalError::TypeError("const_eval_select arg[0] is not a tuple"));
                };
                let layout = self.layout(&tuple.ty)?;
                for (i, field) in fields.iter(Interner).enumerate() {
                    let field = field.assert_ty_ref(Interner).clone();
                    let offset = layout.fields.offset(i).bytes_usize();
                    let addr = tuple.interval.addr.offset(offset);
                    args.push(IntervalAndTy::new(addr, field, self, locals)?);
                }
                self.exec_fn_trait(&args, destination, locals)
            }
            _ => not_supported!("unknown intrinsic {as_str}"),
        }
    }

    fn exec_fn_pointer(
        &mut self,
        bytes: Interval,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals<'_>,
    ) -> Result<()> {
        let id = from_bytes!(usize, bytes.get(self)?);
        let next_ty = self.vtable_map.ty(id)?.clone();
        if let TyKind::FnDef(def, generic_args) = &next_ty.data(Interner).kind {
            self.exec_fn_def(*def, generic_args, destination, args, &locals)?;
        } else {
            return Err(MirEvalError::TypeError("function pointer to non function"));
        }
        Ok(())
    }

    fn exec_closure(
        &mut self,
        closure: ClosureId,
        closure_data: Interval,
        generic_args: &Substitution,
        destination: Interval,
        args: &[IntervalAndTy],
    ) -> Result<()> {
        let mir_body = self
            .db
            .mir_body_for_closure(closure)
            .map_err(|x| MirEvalError::MirLowerErrorForClosure(closure, x))?;
        let arg_bytes = iter::once(Ok(closure_data.get(self)?.to_owned()))
            .chain(args.iter().map(|x| Ok(x.get(&self)?.to_owned())))
            .collect::<Result<Vec<_>>>()?;
        let bytes = self.interpret_mir(&mir_body, arg_bytes.into_iter(), generic_args.clone())?;
        destination.write_from_bytes(self, &bytes)
    }

    fn exec_fn_def(
        &mut self,
        def: FnDefId,
        generic_args: &Substitution,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals<'_>,
    ) -> Result<()> {
        let def: CallableDefId = from_chalk(self.db, def);
        let generic_args = self.subst_filler(generic_args, &locals);
        match def {
            CallableDefId::FunctionId(def) => {
                if let Some(_) = self.detect_fn_trait(def) {
                    self.exec_fn_trait(&args, destination, locals)?;
                    return Ok(());
                }
                self.exec_fn_with_args(def, args, generic_args, locals, destination)?;
            }
            CallableDefId::StructId(id) => {
                let (size, variant_layout, tag) =
                    self.layout_of_variant(id.into(), generic_args, &locals)?;
                let result = self.make_by_layout(
                    size,
                    &variant_layout,
                    tag,
                    args.iter().map(|x| x.interval),
                )?;
                destination.write_from_bytes(self, &result)?;
            }
            CallableDefId::EnumVariantId(id) => {
                let (size, variant_layout, tag) =
                    self.layout_of_variant(id.into(), generic_args, &locals)?;
                let result = self.make_by_layout(
                    size,
                    &variant_layout,
                    tag,
                    args.iter().map(|x| x.interval),
                )?;
                destination.write_from_bytes(self, &result)?;
            }
        }
        Ok(())
    }

    fn exec_fn_with_args(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: Substitution,
        locals: &Locals<'_>,
        destination: Interval,
    ) -> Result<()> {
        let function_data = self.db.function_data(def);
        let is_intrinsic = match &function_data.abi {
            Some(abi) => *abi == Interned::new_str("rust-intrinsic"),
            None => match def.lookup(self.db.upcast()).container {
                hir_def::ItemContainerId::ExternBlockId(block) => {
                    let id = block.lookup(self.db.upcast()).id;
                    id.item_tree(self.db.upcast())[id.value].abi.as_deref()
                        == Some("rust-intrinsic")
                }
                _ => false,
            },
        };
        if is_intrinsic {
            return self.exec_intrinsic(
                function_data.name.as_text().unwrap_or_default().as_str(),
                args,
                generic_args,
                destination,
                &locals,
            );
        }
        let arg_bytes =
            args.iter().map(|x| Ok(x.get(&self)?.to_owned())).collect::<Result<Vec<_>>>()?;
        let result = if let Some(x) = self.detect_lang_function(def) {
            self.exec_lang_item(x, &arg_bytes)?
        } else {
            if let Some(self_ty_idx) =
                is_dyn_method(self.db, self.trait_env.clone(), def, generic_args.clone())
            {
                // In the layout of current possible receiver, which at the moment of writing this code is one of
                // `&T`, `&mut T`, `Box<T>`, `Rc<T>`, `Arc<T>`, and `Pin<P>` where `P` is one of possible receivers,
                // the vtable is exactly in the `[ptr_size..2*ptr_size]` bytes. So we can use it without branching on
                // the type.
                let ty = self
                    .vtable_map
                    .ty_of_bytes(&arg_bytes[0][self.ptr_size()..self.ptr_size() * 2])?;
                let mut args_for_target = args.to_vec();
                args_for_target[0] = IntervalAndTy {
                    interval: args_for_target[0].interval.slice(0..self.ptr_size()),
                    ty: ty.clone(),
                };
                let ty = GenericArgData::Ty(ty.clone()).intern(Interner);
                let generics_for_target = Substitution::from_iter(
                    Interner,
                    generic_args.iter(Interner).enumerate().map(|(i, x)| {
                        if i == self_ty_idx {
                            &ty
                        } else {
                            x
                        }
                    }),
                );
                return self.exec_fn_with_args(
                    def,
                    &args_for_target,
                    generics_for_target,
                    locals,
                    destination,
                );
            }
            let (imp, generic_args) =
                lookup_impl_method(self.db, self.trait_env.clone(), def, generic_args);
            let generic_args = self.subst_filler(&generic_args, &locals);
            let def = imp.into();
            let mir_body =
                self.db.mir_body(def).map_err(|e| MirEvalError::MirLowerError(imp, e))?;
            self.interpret_mir(&mir_body, arg_bytes.iter().cloned(), generic_args)
                .map_err(|e| MirEvalError::InFunction(imp, Box::new(e)))?
        };
        destination.write_from_bytes(self, &result)?;
        Ok(())
    }

    fn exec_fn_trait(
        &mut self,
        args: &[IntervalAndTy],
        destination: Interval,
        locals: &Locals<'_>,
    ) -> Result<()> {
        let func = args.get(0).ok_or(MirEvalError::TypeError("fn trait with no arg"))?;
        let mut func_ty = func.ty.clone();
        let mut func_data = func.interval;
        while let TyKind::Ref(_, _, z) = func_ty.kind(Interner) {
            func_ty = z.clone();
            if matches!(func_ty.kind(Interner), TyKind::Dyn(_)) {
                let id =
                    from_bytes!(usize, &func_data.get(self)?[self.ptr_size()..self.ptr_size() * 2]);
                func_data = func_data.slice(0..self.ptr_size());
                func_ty = self.vtable_map.ty(id)?.clone();
            }
            let size = self.size_of_sized(&func_ty, locals, "self type of fn trait")?;
            func_data = Interval { addr: Address::from_bytes(func_data.get(self)?)?, size };
        }
        match &func_ty.data(Interner).kind {
            TyKind::FnDef(def, subst) => {
                self.exec_fn_def(*def, subst, destination, &args[1..], locals)?;
            }
            TyKind::Function(_) => {
                self.exec_fn_pointer(func_data, destination, &args[1..], locals)?;
            }
            TyKind::Closure(closure, subst) => {
                self.exec_closure(*closure, func_data, subst, destination, &args[1..])?;
            }
            x => not_supported!("Call FnTrait methods with type {x:?}"),
        }
        Ok(())
    }

    fn exec_lang_item(&self, x: LangItem, args: &[Vec<u8>]) -> Result<Vec<u8>> {
        use LangItem::*;
        let mut args = args.iter();
        match x {
            // FIXME: we want to find the panic message from arguments, but it wouldn't work
            // currently even if we do that, since macro expansion of panic related macros
            // is dummy.
            PanicFmt | BeginPanic => Err(MirEvalError::Panic("<format-args>".to_string())),
            SliceLen => {
                let arg = args
                    .next()
                    .ok_or(MirEvalError::TypeError("argument of <[T]>::len() is not provided"))?;
                let ptr_size = arg.len() / 2;
                Ok(arg[ptr_size..].into())
            }
            x => not_supported!("Executing lang item {x:?}"),
        }
    }
}

pub fn pad16(x: &[u8], is_signed: bool) -> [u8; 16] {
    let is_negative = is_signed && x.last().unwrap_or(&0) > &128;
    let fill_with = if is_negative { 255 } else { 0 };
    x.iter()
        .copied()
        .chain(iter::repeat(fill_with))
        .take(16)
        .collect::<Vec<u8>>()
        .try_into()
        .expect("iterator take is not working")
}
