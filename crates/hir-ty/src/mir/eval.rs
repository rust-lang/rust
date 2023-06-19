//! This module provides a MIR interpreter, which is used in const eval.

use std::{borrow::Cow, collections::HashMap, fmt::Write, iter, ops::Range};

use base_db::{CrateId, FileId};
use chalk_ir::Mutability;
use either::Either;
use hir_def::{
    builtin_type::BuiltinType,
    data::adt::{StructFlags, VariantData},
    lang_item::{lang_attr, LangItem},
    layout::{TagEncoding, Variants},
    AdtId, DefWithBodyId, EnumVariantId, FunctionId, HasModule, ItemContainerId, Lookup, StaticId,
    VariantId,
};
use hir_expand::InFile;
use intern::Interned;
use la_arena::ArenaMap;
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::never;
use syntax::{SyntaxNodePtr, TextRange};
use triomphe::Arc;

use crate::{
    consteval::{intern_const_scalar, try_const_usize, ConstEvalError},
    db::HirDatabase,
    display::{ClosureStyle, HirDisplay},
    infer::PointerCast,
    layout::{Layout, LayoutError, RustcEnumVariantIdx},
    mapping::from_chalk,
    method_resolution::{is_dyn_method, lookup_impl_method},
    name, static_lifetime,
    traits::FnTrait,
    utils::{detect_variant_from_bytes, ClosureSubst},
    CallableDefId, ClosureId, Const, ConstScalar, FnDefId, GenericArgData, Interner, MemoryMap,
    Substitution, TraitEnvironment, Ty, TyBuilder, TyExt, TyKind,
};

use super::{
    return_slot, AggregateKind, BinOp, CastKind, LocalId, MirBody, MirLowerError, MirSpan, Operand,
    Place, ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp,
};

mod shim;
#[cfg(test)]
mod tests;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(x) => x,
            Err(_) => return Err(MirEvalError::TypeError(stringify!(mismatched size in constructing $ty))),
        }))
    };
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirEvalError::NotSupported(format!($x)))
    };
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VTableMap {
    ty_to_id: FxHashMap<Ty, usize>,
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

    pub(crate) fn ty(&self, id: usize) -> Result<&Ty> {
        self.id_to_ty.get(id).ok_or(MirEvalError::InvalidVTableId(id))
    }

    fn ty_of_bytes(&self, bytes: &[u8]) -> Result<&Ty> {
        let id = from_bytes!(usize, bytes);
        self.ty(id)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TlsData {
    keys: Vec<u128>,
}

impl TlsData {
    fn create_key(&mut self) -> usize {
        self.keys.push(0);
        self.keys.len() - 1
    }

    fn get_key(&mut self, key: usize) -> Result<u128> {
        let r = self.keys.get(key).ok_or_else(|| {
            MirEvalError::UndefinedBehavior(format!("Getting invalid tls key {key}"))
        })?;
        Ok(*r)
    }

    fn set_key(&mut self, key: usize, value: u128) -> Result<()> {
        let r = self.keys.get_mut(key).ok_or_else(|| {
            MirEvalError::UndefinedBehavior(format!("Setting invalid tls key {key}"))
        })?;
        *r = value;
        Ok(())
    }
}

pub struct Evaluator<'a> {
    db: &'a dyn HirDatabase,
    trait_env: Arc<TraitEnvironment>,
    stack: Vec<u8>,
    heap: Vec<u8>,
    /// Stores the global location of the statics. We const evaluate every static first time we need it
    /// and see it's missing, then we add it to this to reuse.
    static_locations: FxHashMap<StaticId, Address>,
    /// We don't really have function pointers, i.e. pointers to some assembly instructions that we can run. Instead, we
    /// store the type as an interned id in place of function and vtable pointers, and we recover back the type at the
    /// time of use.
    vtable_map: VTableMap,
    thread_local_storage: TlsData,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    crate_id: CrateId,
    // FIXME: This is a workaround, see the comment on `interpret_mir`
    assert_placeholder_ty_is_unused: bool,
    /// A general limit on execution, to prevent non terminating programs from breaking r-a main process
    execution_limit: usize,
    /// An additional limit on stack depth, to prevent stack overflow
    stack_depth_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl From<Interval> for IntervalOrOwned {
    fn from(it: Interval) -> IntervalOrOwned {
        IntervalOrOwned::Borrowed(it)
    }
}

impl IntervalOrOwned {
    pub(crate) fn to_vec(self, memory: &Evaluator<'_>) -> Result<Vec<u8>> {
        Ok(match self {
            IntervalOrOwned::Owned(o) => o,
            IntervalOrOwned::Borrowed(b) => b.get(memory)?.to_vec(),
        })
    }

    fn get<'a>(&'a self, memory: &'a Evaluator<'a>) -> Result<&'a [u8]> {
        Ok(match self {
            IntervalOrOwned::Owned(o) => o,
            IntervalOrOwned::Borrowed(b) => b.get(memory)?,
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
    ConstEvalError(String, Box<ConstEvalError>),
    LayoutError(LayoutError, Ty),
    /// Means that code had type errors (or mismatched args) and we shouldn't generate mir in first place.
    TypeError(&'static str),
    /// Means that code had undefined behavior. We don't try to actively detect UB, but if it was detected
    /// then use this type of error.
    UndefinedBehavior(String),
    Panic(String),
    MirLowerError(FunctionId, MirLowerError),
    MirLowerErrorForClosure(ClosureId, MirLowerError),
    TypeIsUnsized(Ty, &'static str),
    NotSupported(String),
    InvalidConst(Const),
    InFunction(Either<FunctionId, ClosureId>, Box<MirEvalError>, MirSpan, DefWithBodyId),
    ExecutionLimitExceeded,
    StackOverflow,
    TargetDataLayoutNotAvailable,
    InvalidVTableId(usize),
    CoerceUnsizedError(Ty),
    LangItemNotFound(LangItem),
}

impl MirEvalError {
    pub fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
    ) -> std::result::Result<(), std::fmt::Error> {
        writeln!(f, "Mir eval error:")?;
        let mut err = self;
        while let MirEvalError::InFunction(func, e, span, def) = err {
            err = e;
            match func {
                Either::Left(func) => {
                    let function_name = db.function_data(*func);
                    writeln!(
                        f,
                        "In function {} ({:?})",
                        function_name.name.display(db.upcast()),
                        func
                    )?;
                }
                Either::Right(clos) => {
                    writeln!(f, "In {:?}", clos)?;
                }
            }
            let source_map = db.body_with_source_map(*def).1;
            let span: InFile<SyntaxNodePtr> = match span {
                MirSpan::ExprId(e) => match source_map.expr_syntax(*e) {
                    Ok(s) => s.map(|x| x.into()),
                    Err(_) => continue,
                },
                MirSpan::PatId(p) => match source_map.pat_syntax(*p) {
                    Ok(s) => s.map(|x| match x {
                        Either::Left(e) => e.into(),
                        Either::Right(e) => e.into(),
                    }),
                    Err(_) => continue,
                },
                MirSpan::Unknown => continue,
            };
            let file_id = span.file_id.original_file(db.upcast());
            let text_range = span.value.text_range();
            writeln!(f, "{}", span_formatter(file_id, text_range))?;
        }
        match err {
            MirEvalError::InFunction(..) => unreachable!(),
            MirEvalError::LayoutError(err, ty) => {
                write!(
                    f,
                    "Layout for type `{}` is not available due {err:?}",
                    ty.display(db).with_closure_style(ClosureStyle::ClosureWithId).to_string()
                )?;
            }
            MirEvalError::MirLowerError(func, err) => {
                let function_name = db.function_data(*func);
                writeln!(
                    f,
                    "MIR lowering for function `{}` ({:?}) failed due:",
                    function_name.name.display(db.upcast()),
                    func
                )?;
                err.pretty_print(f, db, span_formatter)?;
            }
            MirEvalError::ConstEvalError(name, err) => {
                MirLowerError::ConstEvalError(name.clone(), err.clone()).pretty_print(
                    f,
                    db,
                    span_formatter,
                )?;
            }
            MirEvalError::TypeError(_)
            | MirEvalError::UndefinedBehavior(_)
            | MirEvalError::Panic(_)
            | MirEvalError::MirLowerErrorForClosure(_, _)
            | MirEvalError::TypeIsUnsized(_, _)
            | MirEvalError::NotSupported(_)
            | MirEvalError::InvalidConst(_)
            | MirEvalError::ExecutionLimitExceeded
            | MirEvalError::StackOverflow
            | MirEvalError::TargetDataLayoutNotAvailable
            | MirEvalError::CoerceUnsizedError(_)
            | MirEvalError::LangItemNotFound(_)
            | MirEvalError::InvalidVTableId(_) => writeln!(f, "{:?}", err)?,
        }
        Ok(())
    }
}

impl std::fmt::Debug for MirEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstEvalError(arg0, arg1) => {
                f.debug_tuple("ConstEvalError").field(arg0).field(arg1).finish()
            }
            Self::LangItemNotFound(arg0) => f.debug_tuple("LangItemNotFound").field(arg0).finish(),
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
            Self::CoerceUnsizedError(arg0) => {
                f.debug_tuple("CoerceUnsizedError").field(arg0).finish()
            }
            Self::InvalidVTableId(arg0) => f.debug_tuple("InvalidVTableId").field(arg0).finish(),
            Self::NotSupported(arg0) => f.debug_tuple("NotSupported").field(arg0).finish(),
            Self::InvalidConst(arg0) => {
                let data = &arg0.data(Interner);
                f.debug_struct("InvalidConst").field("ty", &data.ty).field("value", &arg0).finish()
            }
            Self::InFunction(func, e, span, _) => {
                let mut e = &**e;
                let mut stack = vec![(*func, *span)];
                while let Self::InFunction(f, next_e, span, _) = e {
                    e = &next_e;
                    stack.push((*f, *span));
                }
                f.debug_struct("WithStack").field("error", e).field("stack", &stack).finish()
            }
        }
    }
}

type Result<T> = std::result::Result<T, MirEvalError>;

#[derive(Debug, Default)]
struct DropFlags {
    need_drop: FxHashSet<Place>,
}

impl DropFlags {
    fn add_place(&mut self, p: Place) {
        if p.iterate_over_parents().any(|x| self.need_drop.contains(&x)) {
            return;
        }
        self.need_drop.retain(|x| !p.is_parent(x));
        self.need_drop.insert(p);
    }

    fn remove_place(&mut self, p: &Place) -> bool {
        // FIXME: replace parents with parts
        self.need_drop.remove(p)
    }
}

#[derive(Debug)]
struct Locals<'a> {
    ptr: &'a ArenaMap<LocalId, Interval>,
    body: &'a MirBody,
    drop_flags: DropFlags,
}

pub fn interpret_mir(
    db: &dyn HirDatabase,
    body: &MirBody,
    // FIXME: This is workaround. Ideally, const generics should have a separate body (issue #7434), but now
    // they share their body with their parent, so in MIR lowering we have locals of the parent body, which
    // might have placeholders. With this argument, we (wrongly) assume that every placeholder type has
    // a zero size, hoping that they are all outside of our current body. Even without a fix for #7434, we can
    // (and probably should) do better here, for example by excluding bindings outside of the target expression.
    assert_placeholder_ty_is_unused: bool,
) -> (Result<Const>, String, String) {
    let ty = body.locals[return_slot()].ty.clone();
    let mut evaluator = Evaluator::new(db, body, assert_placeholder_ty_is_unused);
    let x: Result<Const> = (|| {
        let bytes = evaluator.interpret_mir(&body, None.into_iter())?;
        let mut memory_map = evaluator.create_memory_map(
            &bytes,
            &ty,
            &Locals { ptr: &ArenaMap::new(), body: &body, drop_flags: DropFlags::default() },
        )?;
        memory_map.vtable = evaluator.vtable_map.clone();
        return Ok(intern_const_scalar(ConstScalar::Bytes(bytes, memory_map), ty));
    })();
    (
        x,
        String::from_utf8_lossy(&evaluator.stdout).into_owned(),
        String::from_utf8_lossy(&evaluator.stderr).into_owned(),
    )
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
            thread_local_storage: TlsData::default(),
            static_locations: HashMap::default(),
            db,
            trait_env,
            crate_id,
            stdout: vec![],
            stderr: vec![],
            assert_placeholder_ty_is_unused,
            stack_depth_limit: 100,
            execution_limit: 1000_000,
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
    ) -> Result<(Address, Ty, Option<IntervalOrOwned>)> {
        let mut addr = locals.ptr[p.local].addr;
        let mut ty: Ty = locals.body.locals[p.local].ty.clone();
        let mut metadata: Option<IntervalOrOwned> = None; // locals are always sized
        for proj in &*p.projection {
            let prev_ty = ty.clone();
            ty = proj.projected_ty(
                ty,
                self.db,
                |c, subst, f| {
                    let (def, _) = self.db.lookup_intern_closure(c.into());
                    let infer = self.db.infer(def);
                    let (captures, _) = infer.closure_info(&c);
                    let parent_subst = ClosureSubst(subst).parent_subst();
                    captures
                        .get(f)
                        .expect("broken closure field")
                        .ty
                        .clone()
                        .substitute(Interner, parent_subst)
                },
                self.crate_id,
            );
            match proj {
                ProjectionElem::Deref => {
                    metadata = if self.size_align_of(&ty, locals)?.is_none() {
                        Some(
                            Interval { addr: addr.offset(self.ptr_size()), size: self.ptr_size() }
                                .into(),
                        )
                    } else {
                        None
                    };
                    let x = from_bytes!(usize, self.read_memory(addr, self.ptr_size())?);
                    addr = Address::from_usize(x);
                }
                ProjectionElem::Index(op) => {
                    let offset = from_bytes!(
                        usize,
                        self.read_memory(locals.ptr[*op].addr, self.ptr_size())?
                    );
                    metadata = None; // Result of index is always sized
                    let ty_size =
                        self.size_of_sized(&ty, locals, "array inner type should be sized")?;
                    addr = addr.offset(ty_size * offset);
                }
                &ProjectionElem::ConstantIndex { from_end, offset } => {
                    let offset = if from_end {
                        let len = match prev_ty.kind(Interner) {
                            TyKind::Array(_, c) => match try_const_usize(self.db, c) {
                                Some(x) => x as u64,
                                None => {
                                    not_supported!("indexing array with unknown const from end")
                                }
                            },
                            TyKind::Slice(_) => match metadata {
                                Some(x) => from_bytes!(u64, x.get(self)?),
                                None => not_supported!("slice place without metadata"),
                            },
                            _ => not_supported!("bad type for const index"),
                        };
                        (len - offset - 1) as usize
                    } else {
                        offset as usize
                    };
                    metadata = None; // Result of index is always sized
                    let ty_size =
                        self.size_of_sized(&ty, locals, "array inner type should be sized")?;
                    addr = addr.offset(ty_size * offset);
                }
                &ProjectionElem::Subslice { from, to } => {
                    let inner_ty = match &ty.data(Interner).kind {
                        TyKind::Array(inner, _) | TyKind::Slice(inner) => inner.clone(),
                        _ => TyKind::Error.intern(Interner),
                    };
                    metadata = match metadata {
                        Some(x) => {
                            let prev_len = from_bytes!(u64, x.get(self)?);
                            Some(IntervalOrOwned::Owned(
                                (prev_len - from - to).to_le_bytes().to_vec(),
                            ))
                        }
                        None => None,
                    };
                    let ty_size =
                        self.size_of_sized(&inner_ty, locals, "array inner type should be sized")?;
                    addr = addr.offset(ty_size * (from as usize));
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
                ProjectionElem::OpaqueCast(_) => not_supported!("opaque cast"),
            }
        }
        Ok((addr, ty, metadata))
    }

    fn layout(&self, ty: &Ty) -> Result<Arc<Layout>> {
        self.db
            .layout_of_ty(ty.clone(), self.crate_id)
            .map_err(|e| MirEvalError::LayoutError(e, ty.clone()))
    }

    fn layout_adt(&self, adt: AdtId, subst: Substitution) -> Result<Arc<Layout>> {
        self.db.layout_of_adt(adt, subst.clone(), self.crate_id).map_err(|e| {
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
            &Operand::Static(s) => {
                let ty = self.db.infer(s.into())[self.db.body(s.into()).body_expr].clone();
                TyKind::Ref(Mutability::Not, static_lifetime(), ty).intern(Interner)
            }
        })
    }

    fn operand_ty_and_eval(
        &mut self,
        o: &Operand,
        locals: &mut Locals<'_>,
    ) -> Result<IntervalAndTy> {
        Ok(IntervalAndTy {
            interval: self.eval_operand(o, locals)?,
            ty: self.operand_ty(o, locals)?,
        })
    }

    fn interpret_mir(
        &mut self,
        body: &MirBody,
        args: impl Iterator<Item = Vec<u8>>,
    ) -> Result<Vec<u8>> {
        if let Some(x) = self.stack_depth_limit.checked_sub(1) {
            self.stack_depth_limit = x;
        } else {
            return Err(MirEvalError::StackOverflow);
        }
        let mut current_block_idx = body.start_block;
        let mut locals =
            Locals { ptr: &ArenaMap::new(), body: &body, drop_flags: DropFlags::default() };
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
                    Ok((id, Interval { addr: Stack(my_ptr), size }))
                })
                .collect::<Result<ArenaMap<LocalId, _>>>()?;
            let stack_size = stack_ptr - self.stack.len();
            (addr, stack_size)
        };
        locals.ptr = &locals_ptr;
        self.stack.extend(iter::repeat(0).take(stack_size));
        let mut remain_args = body.param_locals.len();
        for ((l, interval), value) in locals_ptr.iter().skip(1).zip(args) {
            locals.drop_flags.add_place(l.into());
            interval.write_from_bytes(self, &value)?;
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
                        let result = self.eval_rvalue(r, &mut locals)?.to_vec(&self)?;
                        self.write_memory(addr, &result)?;
                        locals.drop_flags.add_place(l.clone());
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
            match &terminator.kind {
                TerminatorKind::Goto { target } => {
                    current_block_idx = *target;
                }
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                    cleanup: _,
                    from_hir_call: _,
                } => {
                    let destination_interval = self.place_interval(destination, &locals)?;
                    let fn_ty = self.operand_ty(func, &locals)?;
                    let args = args
                        .iter()
                        .map(|x| self.operand_ty_and_eval(x, &mut locals))
                        .collect::<Result<Vec<_>>>()?;
                    match &fn_ty.data(Interner).kind {
                        TyKind::Function(_) => {
                            let bytes = self.eval_operand(func, &mut locals)?;
                            self.exec_fn_pointer(
                                bytes,
                                destination_interval,
                                &args,
                                &locals,
                                terminator.span,
                            )?;
                        }
                        TyKind::FnDef(def, generic_args) => {
                            self.exec_fn_def(
                                *def,
                                generic_args,
                                destination_interval,
                                &args,
                                &locals,
                                terminator.span,
                            )?;
                        }
                        x => not_supported!("unknown function type {x:?}"),
                    }
                    locals.drop_flags.add_place(destination.clone());
                    current_block_idx = target.expect("broken mir, function without target");
                }
                TerminatorKind::SwitchInt { discr, targets } => {
                    let val = u128::from_le_bytes(pad16(
                        self.eval_operand(discr, &mut locals)?.get(&self)?,
                        false,
                    ));
                    current_block_idx = targets.target_for_value(val);
                }
                TerminatorKind::Return => {
                    self.stack_depth_limit += 1;
                    return Ok(locals.ptr[return_slot()].get(self)?.to_vec());
                }
                TerminatorKind::Unreachable => {
                    return Err(MirEvalError::UndefinedBehavior("unreachable executed".to_owned()));
                }
                TerminatorKind::Drop { place, target, unwind: _ } => {
                    self.drop_place(place, &mut locals, terminator.span)?;
                    current_block_idx = *target;
                }
                _ => not_supported!("unknown terminator"),
            }
        }
    }

    fn eval_rvalue(&mut self, r: &Rvalue, locals: &mut Locals<'_>) -> Result<IntervalOrOwned> {
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
            Rvalue::Len(p) => {
                let (_, _, metadata) = self.place_addr_and_ty_and_metadata(p, locals)?;
                match metadata {
                    Some(m) => m,
                    None => {
                        return Err(MirEvalError::TypeError(
                            "type without metadata is used for Rvalue::Len",
                        ));
                    }
                }
            }
            Rvalue::UnaryOp(op, val) => {
                let mut c = self.eval_operand(val, locals)?.get(&self)?;
                let mut ty = self.operand_ty(val, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = self.size_of_sized(&ty, locals, "operand of unary op")?;
                    c = self.read_memory(Address::from_bytes(c)?, size)?;
                }
                if let TyKind::Scalar(chalk_ir::Scalar::Float(f)) = ty.kind(Interner) {
                    match f {
                        chalk_ir::FloatTy::F32 => {
                            let c = -from_bytes!(f32, c);
                            Owned(c.to_le_bytes().into())
                        }
                        chalk_ir::FloatTy::F64 => {
                            let c = -from_bytes!(f64, c);
                            Owned(c.to_le_bytes().into())
                        }
                    }
                } else {
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
            }
            Rvalue::CheckedBinaryOp(op, lhs, rhs) => 'binary_op: {
                let lc = self.eval_operand(lhs, locals)?;
                let rc = self.eval_operand(rhs, locals)?;
                let mut lc = lc.get(&self)?;
                let mut rc = rc.get(&self)?;
                let mut ty = self.operand_ty(lhs, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = if ty.kind(Interner) == &TyKind::Str {
                        if *op != BinOp::Eq {
                            never!("Only eq is builtin for `str`");
                        }
                        let ls = from_bytes!(usize, &lc[self.ptr_size()..self.ptr_size() * 2]);
                        let rs = from_bytes!(usize, &rc[self.ptr_size()..self.ptr_size() * 2]);
                        if ls != rs {
                            break 'binary_op Owned(vec![0]);
                        }
                        lc = &lc[..self.ptr_size()];
                        rc = &rc[..self.ptr_size()];
                        ls
                    } else {
                        self.size_of_sized(&ty, locals, "operand of binary op")?
                    };
                    lc = self.read_memory(Address::from_bytes(lc)?, size)?;
                    rc = self.read_memory(Address::from_bytes(rc)?, size)?;
                }
                if let TyKind::Scalar(chalk_ir::Scalar::Float(f)) = ty.kind(Interner) {
                    match f {
                        chalk_ir::FloatTy::F32 => {
                            let l = from_bytes!(f32, lc);
                            let r = from_bytes!(f32, rc);
                            match op {
                                BinOp::Ge
                                | BinOp::Gt
                                | BinOp::Le
                                | BinOp::Lt
                                | BinOp::Eq
                                | BinOp::Ne => {
                                    let r = op.run_compare(l, r) as u8;
                                    Owned(vec![r])
                                }
                                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                                    let r = match op {
                                        BinOp::Add => l + r,
                                        BinOp::Sub => l - r,
                                        BinOp::Mul => l * r,
                                        BinOp::Div => l / r,
                                        _ => unreachable!(),
                                    };
                                    Owned(r.to_le_bytes().into())
                                }
                                x => not_supported!(
                                    "invalid binop {x:?} on floating point operators"
                                ),
                            }
                        }
                        chalk_ir::FloatTy::F64 => {
                            let l = from_bytes!(f64, lc);
                            let r = from_bytes!(f64, rc);
                            match op {
                                BinOp::Ge
                                | BinOp::Gt
                                | BinOp::Le
                                | BinOp::Lt
                                | BinOp::Eq
                                | BinOp::Ne => {
                                    let r = op.run_compare(l, r) as u8;
                                    Owned(vec![r])
                                }
                                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                                    let r = match op {
                                        BinOp::Add => l + r,
                                        BinOp::Sub => l - r,
                                        BinOp::Mul => l * r,
                                        BinOp::Div => l / r,
                                        _ => unreachable!(),
                                    };
                                    Owned(r.to_le_bytes().into())
                                }
                                x => not_supported!(
                                    "invalid binop {x:?} on floating point operators"
                                ),
                            }
                        }
                    }
                } else {
                    let is_signed = matches!(ty.as_builtin(), Some(BuiltinType::Int(_)));
                    let l128 = i128::from_le_bytes(pad16(lc, is_signed));
                    let r128 = i128::from_le_bytes(pad16(rc, is_signed));
                    let check_overflow = |r: i128| {
                        // FIXME: this is not very correct, and only catches the basic cases.
                        let r = r.to_le_bytes();
                        for &k in &r[lc.len()..] {
                            if k != 0 && (k != 255 || !is_signed) {
                                return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                            }
                        }
                        Ok(Owned(r[0..lc.len()].into()))
                    };
                    match op {
                        BinOp::Ge | BinOp::Gt | BinOp::Le | BinOp::Lt | BinOp::Eq | BinOp::Ne => {
                            let r = op.run_compare(l128, r128) as u8;
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
                            check_overflow(r)?
                        }
                        BinOp::Shl | BinOp::Shr => {
                            let r = 'b: {
                                if let Ok(shift_amount) = u32::try_from(r128) {
                                    let r = match op {
                                        BinOp::Shl => l128.checked_shl(shift_amount),
                                        BinOp::Shr => l128.checked_shr(shift_amount),
                                        _ => unreachable!(),
                                    };
                                    if let Some(r) = r {
                                        break 'b r;
                                    }
                                };
                                return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                            };
                            check_overflow(r)?
                        }
                        BinOp::Offset => not_supported!("offset binop"),
                    }
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
                match &layout.variants {
                    Variants::Single { index } => {
                        let r = self.const_eval_discriminant(EnumVariantId {
                            parent: enum_id,
                            local_id: index.0,
                        })?;
                        Owned(r.to_le_bytes().to_vec())
                    }
                    Variants::Multiple { tag, tag_encoding, variants, .. } => {
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
                                let candidate_tag = i128::from_le_bytes(pad16(tag, false))
                                    .wrapping_sub(*niche_start as i128)
                                    as usize;
                                let variant = variants
                                    .iter_enumerated()
                                    .map(|(x, _)| x)
                                    .filter(|x| x != untagged_variant)
                                    .nth(candidate_tag)
                                    .unwrap_or(*untagged_variant)
                                    .0;
                                let result = self.const_eval_discriminant(EnumVariantId {
                                    parent: enum_id,
                                    local_id: variant,
                                })?;
                                Owned(result.to_le_bytes().to_vec())
                            }
                        }
                    }
                }
            }
            Rvalue::Repeat(x, len) => {
                let len = match try_const_usize(self.db, &len) {
                    Some(x) => x as usize,
                    None => not_supported!("non evaluatable array len in repeat Rvalue"),
                };
                let val = self.eval_operand(x, locals)?.get(self)?;
                let size = len * val.len();
                Owned(val.iter().copied().cycle().take(size).collect())
            }
            Rvalue::ShallowInitBox(_, _) => not_supported!("shallow init box"),
            Rvalue::ShallowInitBoxWithAlloc(ty) => {
                let Some((size, align)) = self.size_align_of(ty, locals)? else {
                    not_supported!("unsized box initialization");
                };
                let addr = self.heap_allocate(size, align);
                Owned(addr.to_bytes())
            }
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
                            values.iter().map(|&x| x.into()),
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
                        let (size, variant_layout, tag) =
                            self.layout_of_variant(*x, subst.clone(), locals)?;
                        Owned(self.make_by_layout(
                            size,
                            &variant_layout,
                            tag,
                            values.iter().map(|&x| x.into()),
                        )?)
                    }
                    AggregateKind::Closure(ty) => {
                        let layout = self.layout(&ty)?;
                        Owned(self.make_by_layout(
                            layout.size.bytes_usize(),
                            &layout,
                            None,
                            values.iter().map(|&x| x.into()),
                        )?)
                    }
                }
            }
            Rvalue::Cast(kind, operand, target_ty) => match kind {
                CastKind::Pointer(cast) => match cast {
                    PointerCast::ReifyFnPointer | PointerCast::ClosureFnPointer(_) => {
                        let current_ty = self.operand_ty(operand, locals)?;
                        if let TyKind::FnDef(_, _) | TyKind::Closure(_, _) =
                            &current_ty.data(Interner).kind
                        {
                            let id = self.vtable_map.id(current_ty);
                            let ptr_size = self.ptr_size();
                            Owned(id.to_le_bytes()[0..ptr_size].to_vec())
                        } else {
                            not_supported!(
                                "creating a fn pointer from a non FnDef or Closure type"
                            );
                        }
                    }
                    PointerCast::Unsize => {
                        let current_ty = self.operand_ty(operand, locals)?;
                        let addr = self.eval_operand(operand, locals)?;
                        self.coerce_unsized(addr, &current_ty, target_ty)?
                    }
                    PointerCast::MutToConstPointer | PointerCast::UnsafeFnPointer => {
                        // This is no-op
                        Borrowed(self.eval_operand(operand, locals)?)
                    }
                    PointerCast::ArrayToPointer => {
                        // We should remove the metadata part if the current type is slice
                        Borrowed(self.eval_operand(operand, locals)?.slice(0..self.ptr_size()))
                    }
                },
                CastKind::DynStar => not_supported!("dyn star cast"),
                CastKind::IntToInt
                | CastKind::PointerExposeAddress
                | CastKind::PointerFromExposedAddress => {
                    let current_ty = self.operand_ty(operand, locals)?;
                    let is_signed = match current_ty.kind(Interner) {
                        TyKind::Scalar(s) => match s {
                            chalk_ir::Scalar::Int(_) => true,
                            _ => false,
                        },
                        _ => false,
                    };
                    let current = pad16(self.eval_operand(operand, locals)?.get(&self)?, is_signed);
                    let dest_size =
                        self.size_of_sized(target_ty, locals, "destination of int to int cast")?;
                    Owned(current[0..dest_size].to_vec())
                }
                CastKind::FloatToInt => not_supported!("float to int cast"),
                CastKind::FloatToFloat => not_supported!("float to float cast"),
                CastKind::IntToFloat => not_supported!("float to int cast"),
                CastKind::FnPtrToPtr => not_supported!("fn ptr to ptr cast"),
            },
        })
    }

    fn coerce_unsized_look_through_fields<T>(
        &self,
        ty: &Ty,
        goal: impl Fn(&TyKind) -> Option<T>,
    ) -> Result<T> {
        let kind = ty.kind(Interner);
        if let Some(x) = goal(kind) {
            return Ok(x);
        }
        if let TyKind::Adt(id, subst) = kind {
            if let AdtId::StructId(struct_id) = id.0 {
                let field_types = self.db.field_types(struct_id.into());
                let mut field_types = field_types.iter();
                if let Some(ty) =
                    field_types.next().map(|x| x.1.clone().substitute(Interner, subst))
                {
                    return self.coerce_unsized_look_through_fields(&ty, goal);
                }
            }
        }
        Err(MirEvalError::CoerceUnsizedError(ty.clone()))
    }

    fn coerce_unsized(
        &mut self,
        addr: Interval,
        current_ty: &Ty,
        target_ty: &Ty,
    ) -> Result<IntervalOrOwned> {
        use IntervalOrOwned::*;
        fn for_ptr(x: &TyKind) -> Option<Ty> {
            match x {
                TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => Some(ty.clone()),
                _ => None,
            }
        }
        Ok(match self.coerce_unsized_look_through_fields(target_ty, for_ptr)? {
            ty => match &ty.data(Interner).kind {
                TyKind::Slice(_) => {
                    match self.coerce_unsized_look_through_fields(current_ty, for_ptr)? {
                        ty => match &ty.data(Interner).kind {
                            TyKind::Array(_, size) => {
                                let len = match try_const_usize(self.db, size) {
                                    None => not_supported!(
                                        "unevaluatble len of array in coerce unsized"
                                    ),
                                    Some(x) => x as usize,
                                };
                                let mut r = Vec::with_capacity(16);
                                let addr = addr.get(self)?;
                                r.extend(addr.iter().copied());
                                r.extend(len.to_le_bytes().into_iter());
                                Owned(r)
                            }
                            t => {
                                not_supported!("slice unsizing from non array type {t:?}")
                            }
                        },
                    }
                }
                TyKind::Dyn(_) => match &current_ty.data(Interner).kind {
                    TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => {
                        let vtable = self.vtable_map.id(ty.clone());
                        let mut r = Vec::with_capacity(16);
                        let addr = addr.get(self)?;
                        r.extend(addr.iter().copied());
                        r.extend(vtable.to_le_bytes().into_iter());
                        Owned(r)
                    }
                    _ => not_supported!("dyn unsizing from non pointers"),
                },
                _ => not_supported!("unknown unsized cast"),
            },
        })
    }

    fn layout_of_variant(
        &mut self,
        x: VariantId,
        subst: Substitution,
        locals: &Locals<'_>,
    ) -> Result<(usize, Arc<Layout>, Option<(usize, usize, i128)>)> {
        let adt = x.adt_id();
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let VariantId::EnumVariantId(x) = x {
                if AdtId::from(f.parent) == adt {
                    // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                    // infinite sized type errors) we use a dummy layout
                    let i = self.const_eval_discriminant(x)?;
                    return Ok((16, self.layout(&TyBuilder::unit())?, Some((0, 16, i))));
                }
            }
        }
        let layout = self.layout_adt(adt, subst)?;
        Ok(match &layout.variants {
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
                let mut discriminant = self.const_eval_discriminant(enum_variant_id)?;
                let variant_layout = variants[rustc_enum_variant_idx].clone();
                let have_tag = match tag_encoding {
                    TagEncoding::Direct => true,
                    TagEncoding::Niche { untagged_variant, niche_variants: _, niche_start } => {
                        if *untagged_variant == rustc_enum_variant_idx {
                            false
                        } else {
                            discriminant = (variants
                                .iter_enumerated()
                                .filter(|(x, _)| x != untagged_variant)
                                .position(|(x, _)| x == rustc_enum_variant_idx)
                                .unwrap() as i128)
                                .wrapping_add(*niche_start as i128);
                            true
                        }
                    }
                };
                (
                    layout.size.bytes_usize(),
                    Arc::new(variant_layout),
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
        values: impl Iterator<Item = IntervalOrOwned>,
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

    fn eval_operand(&mut self, x: &Operand, locals: &mut Locals<'_>) -> Result<Interval> {
        Ok(match x {
            Operand::Copy(p) | Operand::Move(p) => {
                locals.drop_flags.remove_place(p);
                self.eval_place(p, locals)?
            }
            Operand::Static(st) => {
                let addr = self.eval_static(*st, locals)?;
                Interval::new(addr, self.ptr_size())
            }
            Operand::Constant(konst) => {
                let data = &konst.data(Interner);
                match &data.value {
                    chalk_ir::ConstValue::BoundVar(_) => not_supported!("bound var constant"),
                    chalk_ir::ConstValue::InferenceVar(_) => {
                        not_supported!("inference var constant")
                    }
                    chalk_ir::ConstValue::Placeholder(_) => not_supported!("placeholder constant"),
                    chalk_ir::ConstValue::Concrete(c) => {
                        self.allocate_const_in_heap(c, &data.ty, locals, konst)?
                    }
                }
            }
        })
    }

    fn allocate_const_in_heap(
        &mut self,
        c: &chalk_ir::ConcreteConst<Interner>,
        ty: &Ty,
        locals: &Locals<'_>,
        konst: &chalk_ir::Const<Interner>,
    ) -> Result<Interval> {
        Ok(match &c.interned {
            ConstScalar::Bytes(v, memory_map) => {
                let mut v: Cow<'_, [u8]> = Cow::Borrowed(v);
                let patch_map = memory_map.transform_addresses(|b| {
                    let addr = self.heap_allocate(b.len(), 1); // FIXME: align is wrong
                    self.write_memory(addr, b)?;
                    Ok(addr.to_usize())
                })?;
                let (size, align) = self.size_align_of(ty, locals)?.unwrap_or((v.len(), 1));
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
                let addr = self.heap_allocate(size, align);
                self.write_memory(addr, &v)?;
                self.patch_addresses(&patch_map, &memory_map.vtable, addr, ty, locals)?;
                Interval::new(addr, size)
            }
            ConstScalar::UnevaluatedConst(..) => {
                not_supported!("unevaluated const present in monomorphized mir");
            }
            ConstScalar::Unknown => not_supported!("evaluating unknown const"),
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
        if size == 0 {
            return Ok(&[]);
        }
        let (mem, pos) = match addr {
            Stack(x) => (&self.stack, x),
            Heap(x) => (&self.heap, x),
            Invalid(x) => {
                return Err(MirEvalError::UndefinedBehavior(format!(
                    "read invalid memory address {x} with size {size}"
                )));
            }
        };
        mem.get(pos..pos + size)
            .ok_or_else(|| MirEvalError::UndefinedBehavior("out of bound memory read".to_string()))
    }

    fn write_memory(&mut self, addr: Address, r: &[u8]) -> Result<()> {
        if r.is_empty() {
            return Ok(());
        }
        let (mem, pos) = match addr {
            Stack(x) => (&mut self.stack, x),
            Heap(x) => (&mut self.heap, x),
            Invalid(x) => {
                return Err(MirEvalError::UndefinedBehavior(format!(
                    "write invalid memory address {x} with content {r:?}"
                )));
            }
        };
        mem.get_mut(pos..pos + r.len())
            .ok_or_else(|| {
                MirEvalError::UndefinedBehavior("out of bound memory write".to_string())
            })?
            .copy_from_slice(r);
        Ok(())
    }

    fn size_align_of(&self, ty: &Ty, locals: &Locals<'_>) -> Result<Option<(usize, usize)>> {
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let Some((adt, _)) = ty.as_adt() {
                if AdtId::from(f.parent) == adt {
                    // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                    // infinite sized type errors) we use a dummy size
                    return Ok(Some((16, 16)));
                }
            }
        }
        let layout = self.layout(ty);
        if self.assert_placeholder_ty_is_unused {
            if matches!(layout, Err(MirEvalError::LayoutError(LayoutError::HasPlaceholder, _))) {
                return Ok(Some((0, 1)));
            }
        }
        let layout = layout?;
        Ok(layout
            .is_sized()
            .then(|| (layout.size.bytes_usize(), layout.align.abi.bytes() as usize)))
    }

    /// A version of `self.size_of` which returns error if the type is unsized. `what` argument should
    /// be something that complete this: `error: type {ty} was unsized. {what} should be sized`
    fn size_of_sized(&self, ty: &Ty, locals: &Locals<'_>, what: &'static str) -> Result<usize> {
        match self.size_align_of(ty, locals)? {
            Some(x) => Ok(x.0),
            None => Err(MirEvalError::TypeIsUnsized(ty.clone(), what)),
        }
    }

    fn heap_allocate(&mut self, size: usize, _align: usize) -> Address {
        let pos = self.heap.len();
        self.heap.extend(iter::repeat(0).take(size));
        Address::Heap(pos)
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
        fn rec(
            this: &Evaluator<'_>,
            bytes: &[u8],
            ty: &Ty,
            locals: &Locals<'_>,
            mm: &mut MemoryMap,
        ) -> Result<()> {
            match ty.kind(Interner) {
                TyKind::Ref(_, _, t) => {
                    let size = this.size_align_of(t, locals)?;
                    match size {
                        Some((size, _)) => {
                            let addr_usize = from_bytes!(usize, bytes);
                            mm.insert(
                                addr_usize,
                                this.read_memory(Address::from_usize(addr_usize), size)?.to_vec(),
                            )
                        }
                        None => {
                            let mut check_inner = None;
                            let (addr, meta) = bytes.split_at(bytes.len() / 2);
                            let element_size = match t.kind(Interner) {
                                TyKind::Str => 1,
                                TyKind::Slice(t) => {
                                    check_inner = Some(t);
                                    this.size_of_sized(t, locals, "slice inner type")?
                                }
                                TyKind::Dyn(_) => {
                                    let t = this.vtable_map.ty_of_bytes(meta)?;
                                    check_inner = Some(t);
                                    this.size_of_sized(t, locals, "dyn concrete type")?
                                }
                                _ => return Ok(()),
                            };
                            let count = match t.kind(Interner) {
                                TyKind::Dyn(_) => 1,
                                _ => from_bytes!(usize, meta),
                            };
                            let size = element_size * count;
                            let addr = Address::from_bytes(addr)?;
                            let b = this.read_memory(addr, size)?;
                            mm.insert(addr.to_usize(), b.to_vec());
                            if let Some(ty) = check_inner {
                                for i in 0..count {
                                    let offset = element_size * i;
                                    rec(this, &b[offset..offset + element_size], &ty, locals, mm)?;
                                }
                            }
                        }
                    }
                }
                chalk_ir::TyKind::Tuple(_, subst) => {
                    let layout = this.layout(ty)?;
                    for (id, ty) in subst.iter(Interner).enumerate() {
                        let ty = ty.assert_ty_ref(Interner); // Tuple only has type argument
                        let offset = layout.fields.offset(id).bytes_usize();
                        let size = this.layout(ty)?.size.bytes_usize();
                        rec(this, &bytes[offset..offset + size], ty, locals, mm)?;
                    }
                }
                chalk_ir::TyKind::Adt(adt, subst) => match adt.0 {
                    AdtId::StructId(s) => {
                        let data = this.db.struct_data(s);
                        let layout = this.layout(ty)?;
                        let field_types = this.db.field_types(s.into());
                        for (f, _) in data.variant_data.fields().iter() {
                            let offset = layout
                                .fields
                                .offset(u32::from(f.into_raw()) as usize)
                                .bytes_usize();
                            let ty = &field_types[f].clone().substitute(Interner, subst);
                            let size = this.layout(ty)?.size.bytes_usize();
                            rec(this, &bytes[offset..offset + size], ty, locals, mm)?;
                        }
                    }
                    AdtId::EnumId(e) => {
                        let layout = this.layout(ty)?;
                        if let Some((v, l)) =
                            detect_variant_from_bytes(&layout, this.db, this.crate_id, bytes, e)
                        {
                            let data = &this.db.enum_data(e).variants[v].variant_data;
                            let field_types = this
                                .db
                                .field_types(EnumVariantId { parent: e, local_id: v }.into());
                            for (f, _) in data.fields().iter() {
                                let offset =
                                    l.fields.offset(u32::from(f.into_raw()) as usize).bytes_usize();
                                let ty = &field_types[f].clone().substitute(Interner, subst);
                                let size = this.layout(ty)?.size.bytes_usize();
                                rec(this, &bytes[offset..offset + size], ty, locals, mm)?;
                            }
                        }
                    }
                    AdtId::UnionId(_) => (),
                },
                _ => (),
            }
            Ok(())
        }
        let mut mm = MemoryMap::default();
        rec(self, bytes, ty, locals, &mut mm)?;
        Ok(mm)
    }

    fn patch_addresses(
        &mut self,
        patch_map: &HashMap<usize, usize>,
        old_vtable: &VTableMap,
        addr: Address,
        ty: &Ty,
        locals: &Locals<'_>,
    ) -> Result<()> {
        // FIXME: support indirect references
        let layout = self.layout(ty)?;
        let my_size = self.size_of_sized(ty, locals, "value to patch address")?;
        match ty.kind(Interner) {
            TyKind::Ref(_, _, t) => {
                let size = self.size_align_of(t, locals)?;
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
            TyKind::Function(_) => {
                let ty = old_vtable.ty_of_bytes(self.read_memory(addr, my_size)?)?.clone();
                let new_id = self.vtable_map.id(ty);
                self.write_memory(addr, &new_id.to_le_bytes())?;
            }
            TyKind::Adt(id, subst) => match id.0 {
                AdtId::StructId(s) => {
                    for (i, (_, ty)) in self.db.field_types(s.into()).iter().enumerate() {
                        let offset = layout.fields.offset(i).bytes_usize();
                        let ty = ty.clone().substitute(Interner, subst);
                        self.patch_addresses(
                            patch_map,
                            old_vtable,
                            addr.offset(offset),
                            &ty,
                            locals,
                        )?;
                    }
                }
                AdtId::UnionId(_) => (),
                AdtId::EnumId(_) => (),
            },
            TyKind::AssociatedType(_, _)
            | TyKind::Scalar(_)
            | TyKind::Tuple(_, _)
            | TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::Raw(_, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Str
            | TyKind::Never
            | TyKind::Closure(_, _)
            | TyKind::Generator(_, _)
            | TyKind::GeneratorWitness(_, _)
            | TyKind::Foreign(_)
            | TyKind::Error
            | TyKind::Placeholder(_)
            | TyKind::Dyn(_)
            | TyKind::Alias(_)
            | TyKind::BoundVar(_)
            | TyKind::InferenceVar(_, _) => (),
        }
        Ok(())
    }

    fn exec_fn_pointer(
        &mut self,
        bytes: Interval,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals<'_>,
        span: MirSpan,
    ) -> Result<()> {
        let id = from_bytes!(usize, bytes.get(self)?);
        let next_ty = self.vtable_map.ty(id)?.clone();
        match &next_ty.data(Interner).kind {
            TyKind::FnDef(def, generic_args) => {
                self.exec_fn_def(*def, generic_args, destination, args, &locals, span)?;
            }
            TyKind::Closure(id, subst) => {
                self.exec_closure(*id, bytes.slice(0..0), subst, destination, args, locals, span)?;
            }
            _ => return Err(MirEvalError::TypeError("function pointer to non function")),
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
        locals: &Locals<'_>,
        span: MirSpan,
    ) -> Result<()> {
        let mir_body = self
            .db
            .monomorphized_mir_body_for_closure(
                closure,
                generic_args.clone(),
                self.trait_env.clone(),
            )
            .map_err(|x| MirEvalError::MirLowerErrorForClosure(closure, x))?;
        let closure_data = if mir_body.locals[mir_body.param_locals[0]].ty.as_reference().is_some()
        {
            closure_data.addr.to_bytes()
        } else {
            closure_data.get(self)?.to_owned()
        };
        let arg_bytes = iter::once(Ok(closure_data))
            .chain(args.iter().map(|x| Ok(x.get(&self)?.to_owned())))
            .collect::<Result<Vec<_>>>()?;
        let bytes = self.interpret_mir(&mir_body, arg_bytes.into_iter()).map_err(|e| {
            MirEvalError::InFunction(Either::Right(closure), Box::new(e), span, locals.body.owner)
        })?;
        destination.write_from_bytes(self, &bytes)
    }

    fn exec_fn_def(
        &mut self,
        def: FnDefId,
        generic_args: &Substitution,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals<'_>,
        span: MirSpan,
    ) -> Result<()> {
        let def: CallableDefId = from_chalk(self.db, def);
        let generic_args = generic_args.clone();
        match def {
            CallableDefId::FunctionId(def) => {
                if let Some(_) = self.detect_fn_trait(def) {
                    self.exec_fn_trait(&args, destination, locals, span)?;
                    return Ok(());
                }
                self.exec_fn_with_args(def, args, generic_args, locals, destination, span)?;
            }
            CallableDefId::StructId(id) => {
                let (size, variant_layout, tag) =
                    self.layout_of_variant(id.into(), generic_args, &locals)?;
                let result = self.make_by_layout(
                    size,
                    &variant_layout,
                    tag,
                    args.iter().map(|x| x.interval.into()),
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
                    args.iter().map(|x| x.interval.into()),
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
        span: MirSpan,
    ) -> Result<()> {
        if self.detect_and_exec_special_function(
            def,
            args,
            &generic_args,
            locals,
            destination,
            span,
        )? {
            return Ok(());
        }
        let arg_bytes =
            args.iter().map(|x| Ok(x.get(&self)?.to_owned())).collect::<Result<Vec<_>>>()?;
        if let Some(self_ty_idx) =
            is_dyn_method(self.db, self.trait_env.clone(), def, generic_args.clone())
        {
            // In the layout of current possible receiver, which at the moment of writing this code is one of
            // `&T`, `&mut T`, `Box<T>`, `Rc<T>`, `Arc<T>`, and `Pin<P>` where `P` is one of possible recievers,
            // the vtable is exactly in the `[ptr_size..2*ptr_size]` bytes. So we can use it without branching on
            // the type.
            let ty =
                self.vtable_map.ty_of_bytes(&arg_bytes[0][self.ptr_size()..self.ptr_size() * 2])?;
            let mut args_for_target = args.to_vec();
            args_for_target[0] = IntervalAndTy {
                interval: args_for_target[0].interval.slice(0..self.ptr_size()),
                ty: ty.clone(),
            };
            let ty = GenericArgData::Ty(ty.clone()).intern(Interner);
            let generics_for_target =
                Substitution::from_iter(
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
                span,
            );
        }
        let (imp, generic_args) =
            lookup_impl_method(self.db, self.trait_env.clone(), def, generic_args);
        self.exec_looked_up_function(generic_args, locals, imp, arg_bytes, span, destination)
    }

    fn exec_looked_up_function(
        &mut self,
        generic_args: Substitution,
        locals: &Locals<'_>,
        imp: FunctionId,
        arg_bytes: Vec<Vec<u8>>,
        span: MirSpan,
        destination: Interval,
    ) -> Result<()> {
        let def = imp.into();
        let mir_body = self
            .db
            .monomorphized_mir_body(def, generic_args, self.trait_env.clone())
            .map_err(|e| {
                MirEvalError::InFunction(
                    Either::Left(imp),
                    Box::new(MirEvalError::MirLowerError(imp, e)),
                    span,
                    locals.body.owner,
                )
            })?;
        let result = self.interpret_mir(&mir_body, arg_bytes.iter().cloned()).map_err(|e| {
            MirEvalError::InFunction(Either::Left(imp), Box::new(e), span, locals.body.owner)
        })?;
        destination.write_from_bytes(self, &result)?;
        Ok(())
    }

    fn exec_fn_trait(
        &mut self,
        args: &[IntervalAndTy],
        destination: Interval,
        locals: &Locals<'_>,
        span: MirSpan,
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
                self.exec_fn_def(*def, subst, destination, &args[1..], locals, span)?;
            }
            TyKind::Function(_) => {
                self.exec_fn_pointer(func_data, destination, &args[1..], locals, span)?;
            }
            TyKind::Closure(closure, subst) => {
                self.exec_closure(
                    *closure,
                    func_data,
                    &Substitution::from_iter(Interner, ClosureSubst(subst).parent_subst()),
                    destination,
                    &args[1..],
                    locals,
                    span,
                )?;
            }
            x => not_supported!("Call FnTrait methods with type {x:?}"),
        }
        Ok(())
    }

    fn eval_static(&mut self, st: StaticId, locals: &Locals<'_>) -> Result<Address> {
        if let Some(o) = self.static_locations.get(&st) {
            return Ok(*o);
        };
        let static_data = self.db.static_data(st);
        let result = if !static_data.is_extern {
            let konst = self.db.const_eval_static(st).map_err(|e| {
                MirEvalError::ConstEvalError(
                    static_data.name.as_str().unwrap_or("_").to_owned(),
                    Box::new(e),
                )
            })?;
            let data = &konst.data(Interner);
            if let chalk_ir::ConstValue::Concrete(c) = &data.value {
                self.allocate_const_in_heap(&c, &data.ty, locals, &konst)?
            } else {
                not_supported!("unevaluatable static");
            }
        } else {
            let ty = &self.db.infer(st.into())[self.db.body(st.into()).body_expr];
            let Some((size, align)) = self.size_align_of(&ty, locals)? else {
                not_supported!("unsized extern static");
            };
            let addr = self.heap_allocate(size, align);
            Interval::new(addr, size)
        };
        let addr = self.heap_allocate(self.ptr_size(), self.ptr_size());
        self.write_memory(addr, &result.addr.to_bytes())?;
        self.static_locations.insert(st, addr);
        Ok(addr)
    }

    fn const_eval_discriminant(&self, variant: EnumVariantId) -> Result<i128> {
        let r = self.db.const_eval_discriminant(variant);
        match r {
            Ok(r) => Ok(r),
            Err(e) => {
                let data = self.db.enum_data(variant.parent);
                let name = format!(
                    "{}::{}",
                    data.name.display(self.db.upcast()),
                    data.variants[variant.local_id].name.display(self.db.upcast())
                );
                Err(MirEvalError::ConstEvalError(name, Box::new(e)))
            }
        }
    }

    fn drop_place(&mut self, place: &Place, locals: &mut Locals<'_>, span: MirSpan) -> Result<()> {
        let (addr, ty, metadata) = self.place_addr_and_ty_and_metadata(place, locals)?;
        if !locals.drop_flags.remove_place(place) {
            return Ok(());
        }
        let metadata = match metadata {
            Some(x) => x.get(self)?.to_vec(),
            None => vec![],
        };
        self.run_drop_glue_deep(ty, locals, addr, &metadata, span)
    }

    fn run_drop_glue_deep(
        &mut self,
        ty: Ty,
        locals: &Locals<'_>,
        addr: Address,
        _metadata: &[u8],
        span: MirSpan,
    ) -> Result<()> {
        let Some(drop_fn) = (|| {
            let drop_trait = self.db.lang_item(self.crate_id, LangItem::Drop)?.as_trait()?;
            self.db.trait_data(drop_trait).method_by_name(&name![drop])
        })() else {
            // in some tests we don't have drop trait in minicore, and
            // we can ignore drop in them.
            return Ok(());
        };
        let (impl_drop_candidate, subst) = lookup_impl_method(
            self.db,
            self.trait_env.clone(),
            drop_fn,
            Substitution::from1(Interner, ty.clone()),
        );
        if impl_drop_candidate != drop_fn {
            self.exec_looked_up_function(
                subst,
                locals,
                impl_drop_candidate,
                vec![addr.to_bytes()],
                span,
                Interval { addr: Address::Invalid(0), size: 0 },
            )?;
        }
        match ty.kind(Interner) {
            TyKind::Adt(id, subst) => {
                match id.0 {
                    AdtId::StructId(s) => {
                        let data = self.db.struct_data(s);
                        if data.flags.contains(StructFlags::IS_MANUALLY_DROP) {
                            return Ok(());
                        }
                        let layout = self.layout_adt(id.0, subst.clone())?;
                        match data.variant_data.as_ref() {
                            VariantData::Record(fields) | VariantData::Tuple(fields) => {
                                let field_types = self.db.field_types(s.into());
                                for (field, _) in fields.iter() {
                                    let offset = layout
                                        .fields
                                        .offset(u32::from(field.into_raw()) as usize)
                                        .bytes_usize();
                                    let addr = addr.offset(offset);
                                    let ty = field_types[field].clone().substitute(Interner, subst);
                                    self.run_drop_glue_deep(ty, locals, addr, &[], span)?;
                                }
                            }
                            VariantData::Unit => (),
                        }
                    }
                    AdtId::UnionId(_) => (), // union fields don't need drop
                    AdtId::EnumId(_) => (),
                }
            }
            TyKind::AssociatedType(_, _)
            | TyKind::Scalar(_)
            | TyKind::Tuple(_, _)
            | TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::Raw(_, _)
            | TyKind::Ref(_, _, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Str
            | TyKind::Never
            | TyKind::Closure(_, _)
            | TyKind::Generator(_, _)
            | TyKind::GeneratorWitness(_, _)
            | TyKind::Foreign(_)
            | TyKind::Error
            | TyKind::Placeholder(_)
            | TyKind::Dyn(_)
            | TyKind::Alias(_)
            | TyKind::Function(_)
            | TyKind::BoundVar(_)
            | TyKind::InferenceVar(_, _) => (),
        };
        Ok(())
    }

    fn write_to_stdout(&mut self, interval: Interval) -> Result<()> {
        self.stdout.extend(interval.get(self)?.to_vec());
        Ok(())
    }

    fn write_to_stderr(&mut self, interval: Interval) -> Result<()> {
        self.stderr.extend(interval.get(self)?.to_vec());
        Ok(())
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
