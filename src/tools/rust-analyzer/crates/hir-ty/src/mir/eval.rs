//! This module provides a MIR interpreter, which is used in const eval.

use std::{borrow::Cow, cell::RefCell, fmt::Write, iter, mem, ops::Range};

use base_db::Crate;
use chalk_ir::{Mutability, cast::Cast};
use either::Either;
use hir_def::{
    AdtId, DefWithBodyId, EnumVariantId, FunctionId, HasModule, ItemContainerId, Lookup, StaticId,
    VariantId,
    builtin_type::BuiltinType,
    expr_store::HygieneId,
    item_tree::FieldsShape,
    lang_item::LangItem,
    layout::{TagEncoding, Variants},
    resolver::{HasResolver, TypeNs, ValueNs},
    signatures::{StaticFlags, StructFlags},
};
use hir_expand::{InFile, mod_path::path, name::Name};
use intern::sym;
use la_arena::ArenaMap;
use rustc_abi::TargetDataLayout;
use rustc_apfloat::{
    Float,
    ieee::{Half as f16, Quad as f128},
};
use rustc_hash::{FxHashMap, FxHashSet};
use span::FileId;
use stdx::never;
use syntax::{SyntaxNodePtr, TextRange};
use triomphe::Arc;

use crate::{
    CallableDefId, ClosureId, ComplexMemoryMap, Const, ConstData, ConstScalar, FnDefId, Interner,
    MemoryMap, Substitution, ToChalk, TraitEnvironment, Ty, TyBuilder, TyExt, TyKind,
    consteval::{ConstEvalError, intern_const_scalar, try_const_usize},
    db::{HirDatabase, InternedClosure},
    display::{ClosureStyle, DisplayTarget, HirDisplay},
    infer::PointerCast,
    layout::{Layout, LayoutError, RustcEnumVariantIdx},
    mapping::from_chalk,
    method_resolution::{is_dyn_method, lookup_impl_const},
    static_lifetime,
    traits::FnTrait,
    utils::{ClosureSubst, detect_variant_from_bytes},
};

use super::{
    AggregateKind, BasicBlockId, BinOp, CastKind, LocalId, MirBody, MirLowerError, MirSpan,
    Operand, OperandKind, Place, PlaceElem, ProjectionElem, ProjectionStore, Rvalue, StatementKind,
    TerminatorKind, UnOp, return_slot,
};

mod shim;
#[cfg(test)]
mod tests;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            Err(_) => return Err(MirEvalError::InternalError(stringify!(mismatched size in constructing $ty).into())),
        }))
    };
    ($apfloat:tt, $bits:tt, $value:expr) => {
        // FIXME(#17451): Switch to builtin `f16` and `f128` once they are stable.
        $apfloat::from_bits($bits::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            Err(_) => return Err(MirEvalError::InternalError(stringify!(mismatched size in constructing $apfloat).into())),
        }).into())
    };
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirEvalError::NotSupported(format!($it)))
    };
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VTableMap {
    ty_to_id: FxHashMap<Ty, usize>,
    id_to_ty: Vec<Ty>,
}

impl VTableMap {
    const OFFSET: usize = 1000; // We should add some offset to ids to make 0 (null) an invalid id.

    fn id(&mut self, ty: Ty) -> usize {
        if let Some(it) = self.ty_to_id.get(&ty) {
            return *it;
        }
        let id = self.id_to_ty.len() + VTableMap::OFFSET;
        self.id_to_ty.push(ty.clone());
        self.ty_to_id.insert(ty, id);
        id
    }

    pub(crate) fn ty(&self, id: usize) -> Result<&Ty> {
        id.checked_sub(VTableMap::OFFSET)
            .and_then(|id| self.id_to_ty.get(id))
            .ok_or(MirEvalError::InvalidVTableId(id))
    }

    fn ty_of_bytes(&self, bytes: &[u8]) -> Result<&Ty> {
        let id = from_bytes!(usize, bytes);
        self.ty(id)
    }

    pub fn shrink_to_fit(&mut self) {
        self.id_to_ty.shrink_to_fit();
        self.ty_to_id.shrink_to_fit();
    }

    fn is_empty(&self) -> bool {
        self.id_to_ty.is_empty() && self.ty_to_id.is_empty()
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

struct StackFrame {
    locals: Locals,
    destination: Option<BasicBlockId>,
    prev_stack_ptr: usize,
    span: (MirSpan, DefWithBodyId),
}

#[derive(Clone)]
enum MirOrDynIndex {
    Mir(Arc<MirBody>),
    Dyn(usize),
}

pub struct Evaluator<'a> {
    db: &'a dyn HirDatabase,
    trait_env: Arc<TraitEnvironment>,
    target_data_layout: Arc<TargetDataLayout>,
    stack: Vec<u8>,
    heap: Vec<u8>,
    code_stack: Vec<StackFrame>,
    /// Stores the global location of the statics. We const evaluate every static first time we need it
    /// and see it's missing, then we add it to this to reuse.
    static_locations: FxHashMap<StaticId, Address>,
    /// We don't really have function pointers, i.e. pointers to some assembly instructions that we can run. Instead, we
    /// store the type as an interned id in place of function and vtable pointers, and we recover back the type at the
    /// time of use.
    vtable_map: VTableMap,
    thread_local_storage: TlsData,
    random_state: oorandom::Rand64,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    layout_cache: RefCell<FxHashMap<Ty, Arc<Layout>>>,
    projected_ty_cache: RefCell<FxHashMap<(Ty, PlaceElem), Ty>>,
    not_special_fn_cache: RefCell<FxHashSet<FunctionId>>,
    mir_or_dyn_index_cache: RefCell<FxHashMap<(FunctionId, Substitution), MirOrDynIndex>>,
    /// Constantly dropping and creating `Locals` is very costly. We store
    /// old locals that we normally want to drop here, to reuse their allocations
    /// later.
    unused_locals_store: RefCell<FxHashMap<DefWithBodyId, Vec<Locals>>>,
    cached_ptr_size: usize,
    cached_fn_trait_func: Option<FunctionId>,
    cached_fn_mut_trait_func: Option<FunctionId>,
    cached_fn_once_trait_func: Option<FunctionId>,
    crate_id: Crate,
    // FIXME: This is a workaround, see the comment on `interpret_mir`
    assert_placeholder_ty_is_unused: bool,
    /// A general limit on execution, to prevent non terminating programs from breaking r-a main process
    execution_limit: usize,
    /// An additional limit on stack depth, to prevent stack overflow
    stack_depth_limit: usize,
    /// Maximum count of bytes that heap and stack can grow
    memory_limit: usize,
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
        memory.copy_from_interval(self.addr, interval)
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
        locals: &Locals,
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
    fn get<'a>(&'a self, memory: &'a Evaluator<'a>) -> Result<&'a [u8]> {
        Ok(match self {
            IntervalOrOwned::Owned(o) => o,
            IntervalOrOwned::Borrowed(b) => b.get(memory)?,
        })
    }
}

#[cfg(target_pointer_width = "64")]
const STACK_OFFSET: usize = 1 << 60;
#[cfg(target_pointer_width = "64")]
const HEAP_OFFSET: usize = 1 << 59;

#[cfg(target_pointer_width = "32")]
const STACK_OFFSET: usize = 1 << 30;
#[cfg(target_pointer_width = "32")]
const HEAP_OFFSET: usize = 1 << 29;

impl Address {
    #[allow(clippy::double_parens)]
    fn from_bytes(it: &[u8]) -> Result<Self> {
        Ok(Address::from_usize(from_bytes!(usize, it)))
    }

    fn from_usize(it: usize) -> Self {
        if it > STACK_OFFSET {
            Stack(it - STACK_OFFSET)
        } else if it > HEAP_OFFSET {
            Heap(it - HEAP_OFFSET)
        } else {
            Invalid(it)
        }
    }

    fn to_bytes(&self) -> [u8; size_of::<usize>()] {
        usize::to_le_bytes(self.to_usize())
    }

    fn to_usize(&self) -> usize {
        match self {
            Stack(it) => *it + STACK_OFFSET,
            Heap(it) => *it + HEAP_OFFSET,
            Invalid(it) => *it,
        }
    }

    fn map(&self, f: impl FnOnce(usize) -> usize) -> Address {
        match self {
            Stack(it) => Stack(f(*it)),
            Heap(it) => Heap(f(*it)),
            Invalid(it) => Invalid(f(*it)),
        }
    }

    fn offset(&self, offset: usize) -> Address {
        self.map(|it| it + offset)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum MirEvalError {
    ConstEvalError(String, Box<ConstEvalError>),
    LayoutError(LayoutError, Ty),
    TargetDataLayoutNotAvailable(Arc<str>),
    /// Means that code had undefined behavior. We don't try to actively detect UB, but if it was detected
    /// then use this type of error.
    UndefinedBehavior(String),
    Panic(String),
    // FIXME: This should be folded into ConstEvalError?
    MirLowerError(FunctionId, MirLowerError),
    MirLowerErrorForClosure(ClosureId, MirLowerError),
    TypeIsUnsized(Ty, &'static str),
    NotSupported(String),
    InvalidConst(Const),
    InFunction(Box<MirEvalError>, Vec<(Either<FunctionId, ClosureId>, MirSpan, DefWithBodyId)>),
    ExecutionLimitExceeded,
    StackOverflow,
    /// FIXME: Fold this into InternalError
    InvalidVTableId(usize),
    /// ?
    CoerceUnsizedError(Ty),
    /// These should not occur, usually indicates a bug in mir lowering.
    InternalError(Box<str>),
}

impl MirEvalError {
    pub fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        writeln!(f, "Mir eval error:")?;
        let mut err = self;
        while let MirEvalError::InFunction(e, stack) = err {
            err = e;
            for (func, span, def) in stack.iter().take(30).rev() {
                match func {
                    Either::Left(func) => {
                        let function_name = db.function_signature(*func);
                        writeln!(
                            f,
                            "In function {} ({:?})",
                            function_name.name.display(db, display_target.edition),
                            func
                        )?;
                    }
                    Either::Right(closure) => {
                        writeln!(f, "In {closure:?}")?;
                    }
                }
                let source_map = db.body_with_source_map(*def).1;
                let span: InFile<SyntaxNodePtr> = match span {
                    MirSpan::ExprId(e) => match source_map.expr_syntax(*e) {
                        Ok(s) => s.map(|it| it.into()),
                        Err(_) => continue,
                    },
                    MirSpan::PatId(p) => match source_map.pat_syntax(*p) {
                        Ok(s) => s.map(|it| it.syntax_node_ptr()),
                        Err(_) => continue,
                    },
                    MirSpan::BindingId(b) => {
                        match source_map
                            .patterns_for_binding(*b)
                            .iter()
                            .find_map(|p| source_map.pat_syntax(*p).ok())
                        {
                            Some(s) => s.map(|it| it.syntax_node_ptr()),
                            None => continue,
                        }
                    }
                    MirSpan::SelfParam => match source_map.self_param_syntax() {
                        Some(s) => s.map(|it| it.syntax_node_ptr()),
                        None => continue,
                    },
                    MirSpan::Unknown => continue,
                };
                let file_id = span.file_id.original_file(db);
                let text_range = span.value.text_range();
                writeln!(f, "{}", span_formatter(file_id.file_id(db), text_range))?;
            }
        }
        match err {
            MirEvalError::InFunction(..) => unreachable!(),
            MirEvalError::LayoutError(err, ty) => {
                write!(
                    f,
                    "Layout for type `{}` is not available due {err:?}",
                    ty.display(db, display_target).with_closure_style(ClosureStyle::ClosureWithId)
                )?;
            }
            MirEvalError::MirLowerError(func, err) => {
                let function_name = db.function_signature(*func);
                let self_ = match func.lookup(db).container {
                    ItemContainerId::ImplId(impl_id) => Some({
                        let generics = crate::generics::generics(db, impl_id.into());
                        let substs = generics.placeholder_subst(db);
                        db.impl_self_ty(impl_id)
                            .substitute(Interner, &substs)
                            .display(db, display_target)
                            .to_string()
                    }),
                    ItemContainerId::TraitId(it) => Some(
                        db.trait_signature(it).name.display(db, display_target.edition).to_string(),
                    ),
                    _ => None,
                };
                writeln!(
                    f,
                    "MIR lowering for function `{}{}{}` ({:?}) failed due:",
                    self_.as_deref().unwrap_or_default(),
                    if self_.is_some() { "::" } else { "" },
                    function_name.name.display(db, display_target.edition),
                    func
                )?;
                err.pretty_print(f, db, span_formatter, display_target)?;
            }
            MirEvalError::ConstEvalError(name, err) => {
                MirLowerError::ConstEvalError((**name).into(), err.clone()).pretty_print(
                    f,
                    db,
                    span_formatter,
                    display_target,
                )?;
            }
            MirEvalError::UndefinedBehavior(_)
            | MirEvalError::TargetDataLayoutNotAvailable(_)
            | MirEvalError::Panic(_)
            | MirEvalError::MirLowerErrorForClosure(_, _)
            | MirEvalError::TypeIsUnsized(_, _)
            | MirEvalError::NotSupported(_)
            | MirEvalError::InvalidConst(_)
            | MirEvalError::ExecutionLimitExceeded
            | MirEvalError::StackOverflow
            | MirEvalError::CoerceUnsizedError(_)
            | MirEvalError::InternalError(_)
            | MirEvalError::InvalidVTableId(_) => writeln!(f, "{err:?}")?,
        }
        Ok(())
    }

    pub fn is_panic(&self) -> Option<&str> {
        let mut err = self;
        while let MirEvalError::InFunction(e, _) = err {
            err = e;
        }
        match err {
            MirEvalError::Panic(msg) => Some(msg),
            _ => None,
        }
    }
}

impl std::fmt::Debug for MirEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstEvalError(arg0, arg1) => {
                f.debug_tuple("ConstEvalError").field(arg0).field(arg1).finish()
            }
            Self::LayoutError(arg0, arg1) => {
                f.debug_tuple("LayoutError").field(arg0).field(arg1).finish()
            }
            Self::UndefinedBehavior(arg0) => {
                f.debug_tuple("UndefinedBehavior").field(arg0).finish()
            }
            Self::Panic(msg) => write!(f, "Panic with message:\n{msg:?}"),
            Self::TargetDataLayoutNotAvailable(arg0) => {
                f.debug_tuple("TargetDataLayoutNotAvailable").field(arg0).finish()
            }
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
            Self::InternalError(arg0) => f.debug_tuple("InternalError").field(arg0).finish(),
            Self::InvalidVTableId(arg0) => f.debug_tuple("InvalidVTableId").field(arg0).finish(),
            Self::NotSupported(arg0) => f.debug_tuple("NotSupported").field(arg0).finish(),
            Self::InvalidConst(arg0) => {
                let data = &arg0.data(Interner);
                f.debug_struct("InvalidConst").field("ty", &data.ty).field("value", &arg0).finish()
            }
            Self::InFunction(e, stack) => {
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
    fn add_place(&mut self, p: Place, store: &ProjectionStore) {
        if p.iterate_over_parents(store).any(|it| self.need_drop.contains(&it)) {
            return;
        }
        self.need_drop.retain(|it| !p.is_parent(it, store));
        self.need_drop.insert(p);
    }

    fn remove_place(&mut self, p: &Place, store: &ProjectionStore) -> bool {
        // FIXME: replace parents with parts
        if let Some(parent) = p.iterate_over_parents(store).find(|it| self.need_drop.contains(it)) {
            self.need_drop.remove(&parent);
            return true;
        }
        self.need_drop.remove(p)
    }

    fn clear(&mut self) {
        self.need_drop.clear();
    }
}

#[derive(Debug)]
struct Locals {
    ptr: ArenaMap<LocalId, Interval>,
    body: Arc<MirBody>,
    drop_flags: DropFlags,
}

pub struct MirOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl MirOutput {
    pub fn stdout(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.stdout)
    }
    pub fn stderr(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.stderr)
    }
}

pub fn interpret_mir(
    db: &dyn HirDatabase,
    body: Arc<MirBody>,
    // FIXME: This is workaround. Ideally, const generics should have a separate body (issue #7434), but now
    // they share their body with their parent, so in MIR lowering we have locals of the parent body, which
    // might have placeholders. With this argument, we (wrongly) assume that every placeholder type has
    // a zero size, hoping that they are all outside of our current body. Even without a fix for #7434, we can
    // (and probably should) do better here, for example by excluding bindings outside of the target expression.
    assert_placeholder_ty_is_unused: bool,
    trait_env: Option<Arc<TraitEnvironment>>,
) -> Result<(Result<Const>, MirOutput)> {
    let ty = body.locals[return_slot()].ty.clone();
    let mut evaluator = Evaluator::new(db, body.owner, assert_placeholder_ty_is_unused, trait_env)?;
    let it: Result<Const> = (|| {
        if evaluator.ptr_size() != size_of::<usize>() {
            not_supported!("targets with different pointer size from host");
        }
        let interval = evaluator.interpret_mir(body.clone(), None.into_iter())?;
        let bytes = interval.get(&evaluator)?;
        let mut memory_map = evaluator.create_memory_map(
            bytes,
            &ty,
            &Locals { ptr: ArenaMap::new(), body, drop_flags: DropFlags::default() },
        )?;
        let bytes = bytes.into();
        let memory_map = if memory_map.memory.is_empty() && evaluator.vtable_map.is_empty() {
            MemoryMap::Empty
        } else {
            memory_map.vtable = mem::take(&mut evaluator.vtable_map);
            memory_map.vtable.shrink_to_fit();
            MemoryMap::Complex(Box::new(memory_map))
        };
        Ok(intern_const_scalar(ConstScalar::Bytes(bytes, memory_map), ty))
    })();
    Ok((it, MirOutput { stdout: evaluator.stdout, stderr: evaluator.stderr }))
}

#[cfg(test)]
const EXECUTION_LIMIT: usize = 100_000;
#[cfg(not(test))]
const EXECUTION_LIMIT: usize = 10_000_000;

impl Evaluator<'_> {
    pub fn new(
        db: &dyn HirDatabase,
        owner: DefWithBodyId,
        assert_placeholder_ty_is_unused: bool,
        trait_env: Option<Arc<TraitEnvironment>>,
    ) -> Result<Evaluator<'_>> {
        let crate_id = owner.module(db).krate();
        let target_data_layout = match db.target_data_layout(crate_id) {
            Ok(target_data_layout) => target_data_layout,
            Err(e) => return Err(MirEvalError::TargetDataLayoutNotAvailable(e)),
        };
        let cached_ptr_size = target_data_layout.pointer_size.bytes_usize();
        Ok(Evaluator {
            target_data_layout,
            stack: vec![0],
            heap: vec![0],
            code_stack: vec![],
            vtable_map: VTableMap::default(),
            thread_local_storage: TlsData::default(),
            static_locations: Default::default(),
            db,
            random_state: oorandom::Rand64::new(0),
            trait_env: trait_env.unwrap_or_else(|| db.trait_environment_for_body(owner)),
            crate_id,
            stdout: vec![],
            stderr: vec![],
            assert_placeholder_ty_is_unused,
            stack_depth_limit: 100,
            execution_limit: EXECUTION_LIMIT,
            memory_limit: 1_000_000_000, // 2GB, 1GB for stack and 1GB for heap
            layout_cache: RefCell::new(Default::default()),
            projected_ty_cache: RefCell::new(Default::default()),
            not_special_fn_cache: RefCell::new(Default::default()),
            mir_or_dyn_index_cache: RefCell::new(Default::default()),
            unused_locals_store: RefCell::new(Default::default()),
            cached_ptr_size,
            cached_fn_trait_func: LangItem::Fn
                .resolve_trait(db, crate_id)
                .and_then(|x| x.trait_items(db).method_by_name(&Name::new_symbol_root(sym::call))),
            cached_fn_mut_trait_func: LangItem::FnMut.resolve_trait(db, crate_id).and_then(|x| {
                x.trait_items(db).method_by_name(&Name::new_symbol_root(sym::call_mut))
            }),
            cached_fn_once_trait_func: LangItem::FnOnce.resolve_trait(db, crate_id).and_then(|x| {
                x.trait_items(db).method_by_name(&Name::new_symbol_root(sym::call_once))
            }),
        })
    }

    fn place_addr(&self, p: &Place, locals: &Locals) -> Result<Address> {
        Ok(self.place_addr_and_ty_and_metadata(p, locals)?.0)
    }

    fn place_interval(&self, p: &Place, locals: &Locals) -> Result<Interval> {
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
        self.cached_ptr_size
    }

    fn projected_ty(&self, ty: Ty, proj: PlaceElem) -> Ty {
        let pair = (ty, proj);
        if let Some(r) = self.projected_ty_cache.borrow().get(&pair) {
            return r.clone();
        }
        let (ty, proj) = pair;
        let r = proj.projected_ty(
            ty.clone(),
            self.db,
            |c, subst, f| {
                let InternedClosure(def, _) = self.db.lookup_intern_closure(c.into());
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
        self.projected_ty_cache.borrow_mut().insert((ty, proj), r.clone());
        r
    }

    fn place_addr_and_ty_and_metadata<'a>(
        &'a self,
        p: &Place,
        locals: &'a Locals,
    ) -> Result<(Address, Ty, Option<IntervalOrOwned>)> {
        let mut addr = locals.ptr[p.local].addr;
        let mut ty: Ty = locals.body.locals[p.local].ty.clone();
        let mut metadata: Option<IntervalOrOwned> = None; // locals are always sized
        for proj in p.projection.lookup(&locals.body.projection_store) {
            let prev_ty = ty.clone();
            ty = self.projected_ty(ty, proj.clone());
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
                    let it = from_bytes!(usize, self.read_memory(addr, self.ptr_size())?);
                    addr = Address::from_usize(it);
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
                                Some(it) => it as u64,
                                None => {
                                    not_supported!("indexing array with unknown const from end")
                                }
                            },
                            TyKind::Slice(_) => match metadata {
                                Some(it) => from_bytes!(u64, it.get(self)?),
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
                    let inner_ty = match &ty.kind(Interner) {
                        TyKind::Array(inner, _) | TyKind::Slice(inner) => inner.clone(),
                        _ => TyKind::Error.intern(Interner),
                    };
                    metadata = match metadata {
                        Some(it) => {
                            let prev_len = from_bytes!(u64, it.get(self)?);
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
                &ProjectionElem::ClosureField(f) => {
                    let layout = self.layout(&prev_ty)?;
                    let offset = layout.fields.offset(f).bytes_usize();
                    addr = addr.offset(offset);
                    metadata = None;
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let layout = self.layout(&prev_ty)?;
                    let offset = layout.fields.offset(f.index as usize).bytes_usize();
                    addr = addr.offset(offset);
                    metadata = None; // tuple field is always sized FIXME: This is wrong, the tail can be unsized
                }
                ProjectionElem::Field(Either::Left(f)) => {
                    let layout = self.layout(&prev_ty)?;
                    let variant_layout = match &layout.variants {
                        Variants::Single { .. } | Variants::Empty => &layout,
                        Variants::Multiple { variants, .. } => {
                            &variants[match f.parent {
                                hir_def::VariantId::EnumVariantId(it) => {
                                    RustcEnumVariantIdx(it.lookup(self.db).index as usize)
                                }
                                _ => {
                                    return Err(MirEvalError::InternalError(
                                        "mismatched layout".into(),
                                    ));
                                }
                            }]
                        }
                    };
                    let offset = variant_layout
                        .fields
                        .offset(u32::from(f.local_id.into_raw()) as usize)
                        .bytes_usize();
                    addr = addr.offset(offset);
                    // Unsized field metadata is equal to the metadata of the struct
                    if self.size_align_of(&ty, locals)?.is_some() {
                        metadata = None;
                    }
                }
                ProjectionElem::OpaqueCast(_) => not_supported!("opaque cast"),
            }
        }
        Ok((addr, ty, metadata))
    }

    fn layout(&self, ty: &Ty) -> Result<Arc<Layout>> {
        if let Some(x) = self.layout_cache.borrow().get(ty) {
            return Ok(x.clone());
        }
        let r = self
            .db
            .layout_of_ty(ty.clone(), self.trait_env.clone())
            .map_err(|e| MirEvalError::LayoutError(e, ty.clone()))?;
        self.layout_cache.borrow_mut().insert(ty.clone(), r.clone());
        Ok(r)
    }

    fn layout_adt(&self, adt: AdtId, subst: Substitution) -> Result<Arc<Layout>> {
        self.layout(&TyKind::Adt(chalk_ir::AdtId(adt), subst).intern(Interner))
    }

    fn place_ty<'a>(&'a self, p: &Place, locals: &'a Locals) -> Result<Ty> {
        Ok(self.place_addr_and_ty_and_metadata(p, locals)?.1)
    }

    fn operand_ty(&self, o: &Operand, locals: &Locals) -> Result<Ty> {
        Ok(match &o.kind {
            OperandKind::Copy(p) | OperandKind::Move(p) => self.place_ty(p, locals)?,
            OperandKind::Constant(c) => c.data(Interner).ty.clone(),
            &OperandKind::Static(s) => {
                let ty = self.db.infer(s.into())[self.db.body(s.into()).body_expr].clone();
                TyKind::Ref(Mutability::Not, static_lifetime(), ty).intern(Interner)
            }
        })
    }

    fn operand_ty_and_eval(&mut self, o: &Operand, locals: &mut Locals) -> Result<IntervalAndTy> {
        Ok(IntervalAndTy {
            interval: self.eval_operand(o, locals)?,
            ty: self.operand_ty(o, locals)?,
        })
    }

    fn interpret_mir(
        &mut self,
        body: Arc<MirBody>,
        args: impl Iterator<Item = IntervalOrOwned>,
    ) -> Result<Interval> {
        if let Some(it) = self.stack_depth_limit.checked_sub(1) {
            self.stack_depth_limit = it;
        } else {
            return Err(MirEvalError::StackOverflow);
        }
        let mut current_block_idx = body.start_block;
        let (mut locals, prev_stack_ptr) = self.create_locals_for_body(&body, None)?;
        self.fill_locals_for_body(&body, &mut locals, args)?;
        let prev_code_stack = mem::take(&mut self.code_stack);
        let span = (MirSpan::Unknown, body.owner);
        self.code_stack.push(StackFrame { locals, destination: None, prev_stack_ptr, span });
        'stack: loop {
            let Some(mut my_stack_frame) = self.code_stack.pop() else {
                not_supported!("missing stack frame");
            };
            let e = (|| {
                let locals = &mut my_stack_frame.locals;
                let body = locals.body.clone();
                loop {
                    let current_block = &body.basic_blocks[current_block_idx];
                    if let Some(it) = self.execution_limit.checked_sub(1) {
                        self.execution_limit = it;
                    } else {
                        return Err(MirEvalError::ExecutionLimitExceeded);
                    }
                    for statement in &current_block.statements {
                        match &statement.kind {
                            StatementKind::Assign(l, r) => {
                                let addr = self.place_addr(l, locals)?;
                                let result = self.eval_rvalue(r, locals)?;
                                self.copy_from_interval_or_owned(addr, result)?;
                                locals.drop_flags.add_place(*l, &locals.body.projection_store);
                            }
                            StatementKind::Deinit(_) => not_supported!("de-init statement"),
                            StatementKind::StorageLive(_)
                            | StatementKind::FakeRead(_)
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
                            let destination_interval = self.place_interval(destination, locals)?;
                            let fn_ty = self.operand_ty(func, locals)?;
                            let args = args
                                .iter()
                                .map(|it| self.operand_ty_and_eval(it, locals))
                                .collect::<Result<Vec<_>>>()?;
                            let stack_frame = match &fn_ty.kind(Interner) {
                                TyKind::Function(_) => {
                                    let bytes = self.eval_operand(func, locals)?;
                                    self.exec_fn_pointer(
                                        bytes,
                                        destination_interval,
                                        &args,
                                        locals,
                                        *target,
                                        terminator.span,
                                    )?
                                }
                                TyKind::FnDef(def, generic_args) => self.exec_fn_def(
                                    *def,
                                    generic_args,
                                    destination_interval,
                                    &args,
                                    locals,
                                    *target,
                                    terminator.span,
                                )?,
                                it => not_supported!("unknown function type {it:?}"),
                            };
                            locals
                                .drop_flags
                                .add_place(*destination, &locals.body.projection_store);
                            if let Some(stack_frame) = stack_frame {
                                self.code_stack.push(my_stack_frame);
                                current_block_idx = stack_frame.locals.body.start_block;
                                self.code_stack.push(stack_frame);
                                return Ok(None);
                            } else {
                                current_block_idx =
                                    target.ok_or(MirEvalError::UndefinedBehavior(
                                        "Diverging function returned".to_owned(),
                                    ))?;
                            }
                        }
                        TerminatorKind::SwitchInt { discr, targets } => {
                            let val = u128::from_le_bytes(pad16(
                                self.eval_operand(discr, locals)?.get(self)?,
                                false,
                            ));
                            current_block_idx = targets.target_for_value(val);
                        }
                        TerminatorKind::Return => {
                            break;
                        }
                        TerminatorKind::Unreachable => {
                            return Err(MirEvalError::UndefinedBehavior(
                                "unreachable executed".to_owned(),
                            ));
                        }
                        TerminatorKind::Drop { place, target, unwind: _ } => {
                            self.drop_place(place, locals, terminator.span)?;
                            current_block_idx = *target;
                        }
                        _ => not_supported!("unknown terminator"),
                    }
                }
                Ok(Some(my_stack_frame))
            })();
            let my_stack_frame = match e {
                Ok(None) => continue 'stack,
                Ok(Some(x)) => x,
                Err(e) => {
                    let my_code_stack = mem::replace(&mut self.code_stack, prev_code_stack);
                    let mut error_stack = vec![];
                    for frame in my_code_stack.into_iter().rev() {
                        if let DefWithBodyId::FunctionId(f) = frame.locals.body.owner {
                            error_stack.push((Either::Left(f), frame.span.0, frame.span.1));
                        }
                    }
                    return Err(MirEvalError::InFunction(Box::new(e), error_stack));
                }
            };
            let return_interval = my_stack_frame.locals.ptr[return_slot()];
            self.unused_locals_store
                .borrow_mut()
                .entry(my_stack_frame.locals.body.owner)
                .or_default()
                .push(my_stack_frame.locals);
            match my_stack_frame.destination {
                None => {
                    self.code_stack = prev_code_stack;
                    self.stack_depth_limit += 1;
                    return Ok(return_interval);
                }
                Some(bb) => {
                    // We don't support const promotion, so we can't truncate the stack yet.
                    let _ = my_stack_frame.prev_stack_ptr;
                    // self.stack.truncate(my_stack_frame.prev_stack_ptr);
                    current_block_idx = bb;
                }
            }
        }
    }

    fn fill_locals_for_body(
        &mut self,
        body: &MirBody,
        locals: &mut Locals,
        args: impl Iterator<Item = IntervalOrOwned>,
    ) -> Result<()> {
        let mut remain_args = body.param_locals.len();
        for ((l, interval), value) in locals.ptr.iter().skip(1).zip(args) {
            locals.drop_flags.add_place(l.into(), &locals.body.projection_store);
            match value {
                IntervalOrOwned::Owned(value) => interval.write_from_bytes(self, &value)?,
                IntervalOrOwned::Borrowed(value) => interval.write_from_interval(self, value)?,
            }
            if remain_args == 0 {
                return Err(MirEvalError::InternalError("too many arguments".into()));
            }
            remain_args -= 1;
        }
        if remain_args > 0 {
            return Err(MirEvalError::InternalError("too few arguments".into()));
        }
        Ok(())
    }

    fn create_locals_for_body(
        &mut self,
        body: &Arc<MirBody>,
        destination: Option<Interval>,
    ) -> Result<(Locals, usize)> {
        let mut locals =
            match self.unused_locals_store.borrow_mut().entry(body.owner).or_default().pop() {
                None => Locals {
                    ptr: ArenaMap::new(),
                    body: body.clone(),
                    drop_flags: DropFlags::default(),
                },
                Some(mut l) => {
                    l.drop_flags.clear();
                    l.body = body.clone();
                    l
                }
            };
        let stack_size = {
            let mut stack_ptr = self.stack.len();
            for (id, it) in body.locals.iter() {
                if id == return_slot() {
                    if let Some(destination) = destination {
                        locals.ptr.insert(id, destination);
                        continue;
                    }
                }
                let (size, align) = self.size_align_of_sized(
                    &it.ty,
                    &locals,
                    "no unsized local in extending stack",
                )?;
                while stack_ptr % align != 0 {
                    stack_ptr += 1;
                }
                let my_ptr = stack_ptr;
                stack_ptr += size;
                locals.ptr.insert(id, Interval { addr: Stack(my_ptr), size });
            }
            stack_ptr - self.stack.len()
        };
        let prev_stack_pointer = self.stack.len();
        if stack_size > self.memory_limit {
            return Err(MirEvalError::Panic(format!(
                "Stack overflow. Tried to grow stack to {stack_size} bytes"
            )));
        }
        self.stack.extend(std::iter::repeat_n(0, stack_size));
        Ok((locals, prev_stack_pointer))
    }

    fn eval_rvalue(&mut self, r: &Rvalue, locals: &mut Locals) -> Result<IntervalOrOwned> {
        use IntervalOrOwned::*;
        Ok(match r {
            Rvalue::Use(it) => Borrowed(self.eval_operand(it, locals)?),
            Rvalue::Ref(_, p) => {
                let (addr, _, metadata) = self.place_addr_and_ty_and_metadata(p, locals)?;
                let mut r = addr.to_bytes().to_vec();
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
                        return Err(MirEvalError::InternalError(
                            "type without metadata is used for Rvalue::Len".into(),
                        ));
                    }
                }
            }
            Rvalue::UnaryOp(op, val) => {
                let mut c = self.eval_operand(val, locals)?.get(self)?;
                let mut ty = self.operand_ty(val, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = self.size_of_sized(&ty, locals, "operand of unary op")?;
                    c = self.read_memory(Address::from_bytes(c)?, size)?;
                }
                if let TyKind::Scalar(chalk_ir::Scalar::Float(f)) = ty.kind(Interner) {
                    match f {
                        chalk_ir::FloatTy::F16 => {
                            let c = -from_bytes!(f16, u16, c);
                            Owned(u16::try_from(c.to_bits()).unwrap().to_le_bytes().into())
                        }
                        chalk_ir::FloatTy::F32 => {
                            let c = -from_bytes!(f32, c);
                            Owned(c.to_le_bytes().into())
                        }
                        chalk_ir::FloatTy::F64 => {
                            let c = -from_bytes!(f64, c);
                            Owned(c.to_le_bytes().into())
                        }
                        chalk_ir::FloatTy::F128 => {
                            let c = -from_bytes!(f128, u128, c);
                            Owned(c.to_bits().to_le_bytes().into())
                        }
                    }
                } else {
                    let mut c = c.to_vec();
                    if ty.as_builtin() == Some(BuiltinType::Bool) {
                        c[0] = 1 - c[0];
                    } else {
                        match op {
                            UnOp::Not => c.iter_mut().for_each(|it| *it = !*it),
                            UnOp::Neg => {
                                c.iter_mut().for_each(|it| *it = !*it);
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
                let mut lc = lc.get(self)?;
                let mut rc = rc.get(self)?;
                let mut ty = self.operand_ty(lhs, locals)?;
                while let TyKind::Ref(_, _, z) = ty.kind(Interner) {
                    ty = z.clone();
                    let size = if ty.is_str() {
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
                        lc = self.read_memory(Address::from_bytes(lc)?, ls)?;
                        rc = self.read_memory(Address::from_bytes(rc)?, ls)?;
                        break 'binary_op Owned(vec![u8::from(lc == rc)]);
                    } else {
                        self.size_of_sized(&ty, locals, "operand of binary op")?
                    };
                    lc = self.read_memory(Address::from_bytes(lc)?, size)?;
                    rc = self.read_memory(Address::from_bytes(rc)?, size)?;
                }
                if let TyKind::Scalar(chalk_ir::Scalar::Float(f)) = ty.kind(Interner) {
                    match f {
                        chalk_ir::FloatTy::F16 => {
                            let l = from_bytes!(f16, u16, lc);
                            let r = from_bytes!(f16, u16, rc);
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
                                    Owned(
                                        u16::try_from(r.value.to_bits())
                                            .unwrap()
                                            .to_le_bytes()
                                            .into(),
                                    )
                                }
                                it => not_supported!(
                                    "invalid binop {it:?} on floating point operators"
                                ),
                            }
                        }
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
                                it => not_supported!(
                                    "invalid binop {it:?} on floating point operators"
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
                                it => not_supported!(
                                    "invalid binop {it:?} on floating point operators"
                                ),
                            }
                        }
                        chalk_ir::FloatTy::F128 => {
                            let l = from_bytes!(f128, u128, lc);
                            let r = from_bytes!(f128, u128, rc);
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
                                    Owned(r.value.to_bits().to_le_bytes().into())
                                }
                                it => not_supported!(
                                    "invalid binop {it:?} on floating point operators"
                                ),
                            }
                        }
                    }
                } else {
                    let is_signed = matches!(ty.as_builtin(), Some(BuiltinType::Int(_)));
                    let l128 = IntValue::from_bytes(lc, is_signed);
                    let r128 = IntValue::from_bytes(rc, is_signed);
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
                                BinOp::Add => l128.checked_add(r128).ok_or_else(|| {
                                    MirEvalError::Panic(format!("Overflow in {op:?}"))
                                })?,
                                BinOp::Mul => l128.checked_mul(r128).ok_or_else(|| {
                                    MirEvalError::Panic(format!("Overflow in {op:?}"))
                                })?,
                                BinOp::Div => l128.checked_div(r128).ok_or_else(|| {
                                    MirEvalError::Panic(format!("Overflow in {op:?}"))
                                })?,
                                BinOp::Rem => l128.checked_rem(r128).ok_or_else(|| {
                                    MirEvalError::Panic(format!("Overflow in {op:?}"))
                                })?,
                                BinOp::Sub => l128.checked_sub(r128).ok_or_else(|| {
                                    MirEvalError::Panic(format!("Overflow in {op:?}"))
                                })?,
                                BinOp::BitAnd => l128 & r128,
                                BinOp::BitOr => l128 | r128,
                                BinOp::BitXor => l128 ^ r128,
                                _ => unreachable!(),
                            };
                            Owned(r.to_bytes())
                        }
                        BinOp::Shl | BinOp::Shr => {
                            let r = 'b: {
                                if let Some(shift_amount) = r128.as_u32() {
                                    let r = match op {
                                        BinOp::Shl => l128.checked_shl(shift_amount),
                                        BinOp::Shr => l128.checked_shr(shift_amount),
                                        _ => unreachable!(),
                                    };
                                    if shift_amount as usize >= lc.len() * 8 {
                                        return Err(MirEvalError::Panic(format!(
                                            "Overflow in {op:?}"
                                        )));
                                    }
                                    if let Some(r) = r {
                                        break 'b r;
                                    }
                                };
                                return Err(MirEvalError::Panic(format!("Overflow in {op:?}")));
                            };
                            Owned(r.to_bytes())
                        }
                        BinOp::Offset => not_supported!("offset binop"),
                    }
                }
            }
            Rvalue::Discriminant(p) => {
                let ty = self.place_ty(p, locals)?;
                let bytes = self.eval_place(p, locals)?.get(self)?;
                let result = self.compute_discriminant(ty, bytes)?;
                Owned(result.to_le_bytes().to_vec())
            }
            Rvalue::Repeat(it, len) => {
                let len = match try_const_usize(self.db, len) {
                    Some(it) => it as usize,
                    None => not_supported!("non evaluatable array len in repeat Rvalue"),
                };
                let val = self.eval_operand(it, locals)?.get(self)?;
                let size = len * val.len();
                Owned(val.iter().copied().cycle().take(size).collect())
            }
            Rvalue::ShallowInitBox(_, _) => not_supported!("shallow init box"),
            Rvalue::ShallowInitBoxWithAlloc(ty) => {
                let Some((size, align)) = self.size_align_of(ty, locals)? else {
                    not_supported!("unsized box initialization");
                };
                let addr = self.heap_allocate(size, align)?;
                Owned(addr.to_bytes().to_vec())
            }
            Rvalue::CopyForDeref(_) => not_supported!("copy for deref"),
            Rvalue::Aggregate(kind, values) => {
                let values = values
                    .iter()
                    .map(|it| self.eval_operand(it, locals))
                    .collect::<Result<Vec<_>>>()?;
                match kind {
                    AggregateKind::Array(_) => {
                        let mut r = vec![];
                        for it in values {
                            let value = it.get(self)?;
                            r.extend(value);
                        }
                        Owned(r)
                    }
                    AggregateKind::Tuple(ty) => {
                        let layout = self.layout(ty)?;
                        Owned(self.construct_with_layout(
                            layout.size.bytes_usize(),
                            &layout,
                            None,
                            values.iter().map(|&it| it.into()),
                        )?)
                    }
                    AggregateKind::Union(it, f) => {
                        let layout =
                            self.layout_adt((*it).into(), Substitution::empty(Interner))?;
                        let offset = layout
                            .fields
                            .offset(u32::from(f.local_id.into_raw()) as usize)
                            .bytes_usize();
                        let op = values[0].get(self)?;
                        let mut result = vec![0; layout.size.bytes_usize()];
                        result[offset..offset + op.len()].copy_from_slice(op);
                        Owned(result)
                    }
                    AggregateKind::Adt(it, subst) => {
                        let (size, variant_layout, tag) =
                            self.layout_of_variant(*it, subst.clone(), locals)?;
                        Owned(self.construct_with_layout(
                            size,
                            &variant_layout,
                            tag,
                            values.iter().map(|&it| it.into()),
                        )?)
                    }
                    AggregateKind::Closure(ty) => {
                        let layout = self.layout(ty)?;
                        Owned(self.construct_with_layout(
                            layout.size.bytes_usize(),
                            &layout,
                            None,
                            values.iter().map(|&it| it.into()),
                        )?)
                    }
                }
            }
            Rvalue::Cast(kind, operand, target_ty) => match kind {
                CastKind::PointerCoercion(cast) => match cast {
                    PointerCast::ReifyFnPointer | PointerCast::ClosureFnPointer(_) => {
                        let current_ty = self.operand_ty(operand, locals)?;
                        if let TyKind::FnDef(_, _) | TyKind::Closure(_, _) =
                            &current_ty.kind(Interner)
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
                | CastKind::PtrToPtr
                | CastKind::PointerExposeAddress
                | CastKind::PointerFromExposedAddress => {
                    let current_ty = self.operand_ty(operand, locals)?;
                    let is_signed = matches!(
                        current_ty.kind(Interner),
                        TyKind::Scalar(chalk_ir::Scalar::Int(_))
                    );
                    let current = pad16(self.eval_operand(operand, locals)?.get(self)?, is_signed);
                    let dest_size =
                        self.size_of_sized(target_ty, locals, "destination of int to int cast")?;
                    Owned(current[0..dest_size].to_vec())
                }
                CastKind::FloatToInt => {
                    let ty = self.operand_ty(operand, locals)?;
                    let TyKind::Scalar(chalk_ir::Scalar::Float(ty)) = ty.kind(Interner) else {
                        not_supported!("invalid float to int cast");
                    };
                    let value = self.eval_operand(operand, locals)?.get(self)?;
                    let value = match ty {
                        chalk_ir::FloatTy::F32 => {
                            let value = value.try_into().unwrap();
                            f32::from_le_bytes(value) as f64
                        }
                        chalk_ir::FloatTy::F64 => {
                            let value = value.try_into().unwrap();
                            f64::from_le_bytes(value)
                        }
                        chalk_ir::FloatTy::F16 | chalk_ir::FloatTy::F128 => {
                            not_supported!("unstable floating point type f16 and f128");
                        }
                    };
                    let is_signed = matches!(
                        target_ty.kind(Interner),
                        TyKind::Scalar(chalk_ir::Scalar::Int(_))
                    );
                    let dest_size =
                        self.size_of_sized(target_ty, locals, "destination of float to int cast")?;
                    let dest_bits = dest_size * 8;
                    let (max, min) = if dest_bits == 128 {
                        (i128::MAX, i128::MIN)
                    } else if is_signed {
                        let max = 1i128 << (dest_bits - 1);
                        (max - 1, -max)
                    } else {
                        (1i128 << dest_bits, 0)
                    };
                    let value = (value as i128).min(max).max(min);
                    let result = value.to_le_bytes();
                    Owned(result[0..dest_size].to_vec())
                }
                CastKind::FloatToFloat => {
                    let ty = self.operand_ty(operand, locals)?;
                    let TyKind::Scalar(chalk_ir::Scalar::Float(ty)) = ty.kind(Interner) else {
                        not_supported!("invalid float to int cast");
                    };
                    let value = self.eval_operand(operand, locals)?.get(self)?;
                    let value = match ty {
                        chalk_ir::FloatTy::F32 => {
                            let value = value.try_into().unwrap();
                            f32::from_le_bytes(value) as f64
                        }
                        chalk_ir::FloatTy::F64 => {
                            let value = value.try_into().unwrap();
                            f64::from_le_bytes(value)
                        }
                        chalk_ir::FloatTy::F16 | chalk_ir::FloatTy::F128 => {
                            not_supported!("unstable floating point type f16 and f128");
                        }
                    };
                    let TyKind::Scalar(chalk_ir::Scalar::Float(target_ty)) =
                        target_ty.kind(Interner)
                    else {
                        not_supported!("invalid float to float cast");
                    };
                    match target_ty {
                        chalk_ir::FloatTy::F32 => Owned((value as f32).to_le_bytes().to_vec()),
                        chalk_ir::FloatTy::F64 => Owned((value as f64).to_le_bytes().to_vec()),
                        chalk_ir::FloatTy::F16 | chalk_ir::FloatTy::F128 => {
                            not_supported!("unstable floating point type f16 and f128");
                        }
                    }
                }
                CastKind::IntToFloat => {
                    let current_ty = self.operand_ty(operand, locals)?;
                    let is_signed = matches!(
                        current_ty.kind(Interner),
                        TyKind::Scalar(chalk_ir::Scalar::Int(_))
                    );
                    let value = pad16(self.eval_operand(operand, locals)?.get(self)?, is_signed);
                    let value = i128::from_le_bytes(value);
                    let TyKind::Scalar(chalk_ir::Scalar::Float(target_ty)) =
                        target_ty.kind(Interner)
                    else {
                        not_supported!("invalid int to float cast");
                    };
                    match target_ty {
                        chalk_ir::FloatTy::F32 => Owned((value as f32).to_le_bytes().to_vec()),
                        chalk_ir::FloatTy::F64 => Owned((value as f64).to_le_bytes().to_vec()),
                        chalk_ir::FloatTy::F16 | chalk_ir::FloatTy::F128 => {
                            not_supported!("unstable floating point type f16 and f128");
                        }
                    }
                }
                CastKind::FnPtrToPtr => not_supported!("fn ptr to ptr cast"),
            },
            Rvalue::ThreadLocalRef(n)
            | Rvalue::AddressOf(n)
            | Rvalue::BinaryOp(n)
            | Rvalue::NullaryOp(n) => match *n {},
        })
    }

    fn compute_discriminant(&self, ty: Ty, bytes: &[u8]) -> Result<i128> {
        let layout = self.layout(&ty)?;
        let &TyKind::Adt(chalk_ir::AdtId(AdtId::EnumId(e)), _) = ty.kind(Interner) else {
            return Ok(0);
        };
        match &layout.variants {
            Variants::Empty => unreachable!(),
            Variants::Single { index } => {
                let r =
                    self.const_eval_discriminant(e.enum_variants(self.db).variants[index.0].0)?;
                Ok(r)
            }
            Variants::Multiple { tag, tag_encoding, variants, .. } => {
                let size = tag.size(&*self.target_data_layout).bytes_usize();
                let offset = layout.fields.offset(0).bytes_usize(); // The only field on enum variants is the tag field
                let is_signed = tag.is_signed();
                match tag_encoding {
                    TagEncoding::Direct => {
                        let tag = &bytes[offset..offset + size];
                        Ok(i128::from_le_bytes(pad16(tag, is_signed)))
                    }
                    TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                        let tag = &bytes[offset..offset + size];
                        let candidate_tag = i128::from_le_bytes(pad16(tag, is_signed))
                            .wrapping_sub(*niche_start as i128)
                            as usize;
                        let idx = variants
                            .iter_enumerated()
                            .map(|(it, _)| it)
                            .filter(|it| it != untagged_variant)
                            .nth(candidate_tag)
                            .unwrap_or(*untagged_variant)
                            .0;
                        let result =
                            self.const_eval_discriminant(e.enum_variants(self.db).variants[idx].0)?;
                        Ok(result)
                    }
                }
            }
        }
    }

    fn coerce_unsized_look_through_fields<T>(
        &self,
        ty: &Ty,
        goal: impl Fn(&TyKind) -> Option<T>,
    ) -> Result<T> {
        let kind = ty.kind(Interner);
        if let Some(it) = goal(kind) {
            return Ok(it);
        }
        if let TyKind::Adt(id, subst) = kind {
            if let AdtId::StructId(struct_id) = id.0 {
                let field_types = self.db.field_types(struct_id.into());
                if let Some(ty) =
                    field_types.iter().last().map(|it| it.1.clone().substitute(Interner, subst))
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
        fn for_ptr(it: &TyKind) -> Option<Ty> {
            match it {
                TyKind::Raw(_, ty) | TyKind::Ref(_, _, ty) => Some(ty.clone()),
                _ => None,
            }
        }
        let target_ty = self.coerce_unsized_look_through_fields(target_ty, for_ptr)?;
        let current_ty = self.coerce_unsized_look_through_fields(current_ty, for_ptr)?;

        self.unsizing_ptr_from_addr(target_ty, current_ty, addr)
    }

    /// Adds metadata to the address and create the fat pointer result of the unsizing operation.
    fn unsizing_ptr_from_addr(
        &mut self,
        target_ty: Ty,
        current_ty: Ty,
        addr: Interval,
    ) -> Result<IntervalOrOwned> {
        use IntervalOrOwned::*;
        Ok(match &target_ty.kind(Interner) {
            TyKind::Slice(_) => match &current_ty.kind(Interner) {
                TyKind::Array(_, size) => {
                    let len = match try_const_usize(self.db, size) {
                        None => {
                            not_supported!("unevaluatble len of array in coerce unsized")
                        }
                        Some(it) => it as usize,
                    };
                    let mut r = Vec::with_capacity(16);
                    let addr = addr.get(self)?;
                    r.extend(addr.iter().copied());
                    r.extend(len.to_le_bytes());
                    Owned(r)
                }
                t => {
                    not_supported!("slice unsizing from non array type {t:?}")
                }
            },
            TyKind::Dyn(_) => {
                let vtable = self.vtable_map.id(current_ty);
                let mut r = Vec::with_capacity(16);
                let addr = addr.get(self)?;
                r.extend(addr.iter().copied());
                r.extend(vtable.to_le_bytes());
                Owned(r)
            }
            TyKind::Adt(id, target_subst) => match &current_ty.kind(Interner) {
                TyKind::Adt(current_id, current_subst) => {
                    if id != current_id {
                        not_supported!("unsizing struct with different type");
                    }
                    let id = match id.0 {
                        AdtId::StructId(s) => s,
                        AdtId::UnionId(_) => not_supported!("unsizing unions"),
                        AdtId::EnumId(_) => not_supported!("unsizing enums"),
                    };
                    let Some((last_field, _)) = id.fields(self.db).fields().iter().next_back()
                    else {
                        not_supported!("unsizing struct without field");
                    };
                    let target_last_field = self.db.field_types(id.into())[last_field]
                        .clone()
                        .substitute(Interner, target_subst);
                    let current_last_field = self.db.field_types(id.into())[last_field]
                        .clone()
                        .substitute(Interner, current_subst);
                    return self.unsizing_ptr_from_addr(
                        target_last_field,
                        current_last_field,
                        addr,
                    );
                }
                _ => not_supported!("unsizing struct with non adt type"),
            },
            _ => not_supported!("unknown unsized cast"),
        })
    }

    fn layout_of_variant(
        &mut self,
        it: VariantId,
        subst: Substitution,
        locals: &Locals,
    ) -> Result<(usize, Arc<Layout>, Option<(usize, usize, i128)>)> {
        let adt = it.adt_id(self.db);
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let VariantId::EnumVariantId(it) = it {
                if let AdtId::EnumId(e) = adt {
                    if f.lookup(self.db).parent == e {
                        // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                        // infinite sized type errors) we use a dummy layout
                        let i = self.const_eval_discriminant(it)?;
                        return Ok((16, self.layout(&TyBuilder::unit())?, Some((0, 16, i))));
                    }
                }
            }
        }
        let layout = self.layout_adt(adt, subst)?;
        Ok(match &layout.variants {
            Variants::Single { .. } | Variants::Empty => (layout.size.bytes_usize(), layout, None),
            Variants::Multiple { variants, tag, tag_encoding, .. } => {
                let enum_variant_id = match it {
                    VariantId::EnumVariantId(it) => it,
                    _ => not_supported!("multi variant layout for non-enums"),
                };
                let mut discriminant = self.const_eval_discriminant(enum_variant_id)?;
                let lookup = enum_variant_id.lookup(self.db);
                let rustc_enum_variant_idx = RustcEnumVariantIdx(lookup.index as usize);
                let variant_layout = variants[rustc_enum_variant_idx].clone();
                let have_tag = match tag_encoding {
                    TagEncoding::Direct => true,
                    TagEncoding::Niche { untagged_variant, niche_variants: _, niche_start } => {
                        if *untagged_variant == rustc_enum_variant_idx {
                            false
                        } else {
                            discriminant = (variants
                                .iter_enumerated()
                                .filter(|(it, _)| it != untagged_variant)
                                .position(|(it, _)| it == rustc_enum_variant_idx)
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
                            tag.size(&*self.target_data_layout).bytes_usize(),
                            discriminant,
                        ))
                    } else {
                        None
                    },
                )
            }
        })
    }

    fn construct_with_layout(
        &mut self,
        size: usize, // Not necessarily equal to variant_layout.size
        variant_layout: &Layout,
        tag: Option<(usize, usize, i128)>,
        values: impl Iterator<Item = IntervalOrOwned>,
    ) -> Result<Vec<u8>> {
        let mut result = vec![0; size];
        if let Some((offset, size, value)) = tag {
            match result.get_mut(offset..offset + size) {
                Some(it) => it.copy_from_slice(&value.to_le_bytes()[0..size]),
                None => {
                    return Err(MirEvalError::InternalError(
                        format!(
                            "encoded tag ({offset}, {size}, {value}) is out of bounds 0..{size}"
                        )
                        .into(),
                    ));
                }
            }
        }
        for (i, op) in values.enumerate() {
            let offset = variant_layout.fields.offset(i).bytes_usize();
            let op = op.get(self)?;
            match result.get_mut(offset..offset + op.len()) {
                Some(it) => it.copy_from_slice(op),
                None => {
                    return Err(MirEvalError::InternalError(
                        format!("field offset ({offset}) is out of bounds 0..{size}").into(),
                    ));
                }
            }
        }
        Ok(result)
    }

    fn eval_operand(&mut self, it: &Operand, locals: &mut Locals) -> Result<Interval> {
        Ok(match &it.kind {
            OperandKind::Copy(p) | OperandKind::Move(p) => {
                locals.drop_flags.remove_place(p, &locals.body.projection_store);
                self.eval_place(p, locals)?
            }
            OperandKind::Static(st) => {
                let addr = self.eval_static(*st, locals)?;
                Interval::new(addr, self.ptr_size())
            }
            OperandKind::Constant(konst) => self.allocate_const_in_heap(locals, konst)?,
        })
    }

    #[allow(clippy::double_parens)]
    fn allocate_const_in_heap(&mut self, locals: &Locals, konst: &Const) -> Result<Interval> {
        let ConstData { ty, value: chalk_ir::ConstValue::Concrete(c) } = &konst.data(Interner)
        else {
            not_supported!("evaluating non concrete constant");
        };
        let result_owner;
        let (v, memory_map) = match &c.interned {
            ConstScalar::Bytes(v, mm) => (v, mm),
            ConstScalar::UnevaluatedConst(const_id, subst) => 'b: {
                let mut const_id = *const_id;
                let mut subst = subst.clone();
                if let hir_def::GeneralConstId::ConstId(c) = const_id {
                    let (c, s) = lookup_impl_const(self.db, self.trait_env.clone(), c, subst);
                    const_id = hir_def::GeneralConstId::ConstId(c);
                    subst = s;
                }
                result_owner = self
                    .db
                    .const_eval(const_id, subst, Some(self.trait_env.clone()))
                    .map_err(|e| {
                        let name = const_id.name(self.db);
                        MirEvalError::ConstEvalError(name, Box::new(e))
                    })?;
                if let chalk_ir::ConstValue::Concrete(c) = &result_owner.data(Interner).value {
                    if let ConstScalar::Bytes(v, mm) = &c.interned {
                        break 'b (v, mm);
                    }
                }
                not_supported!("unevaluatable constant");
            }
            ConstScalar::Unknown => not_supported!("evaluating unknown const"),
        };
        let patch_map = memory_map.transform_addresses(|b, align| {
            let addr = self.heap_allocate(b.len(), align)?;
            self.write_memory(addr, b)?;
            Ok(addr.to_usize())
        })?;
        let (size, align) = self.size_align_of(ty, locals)?.unwrap_or((v.len(), 1));
        let v: Cow<'_, [u8]> = if size != v.len() {
            // Handle self enum
            if size == 16 && v.len() < 16 {
                Cow::Owned(pad16(v, false).to_vec())
            } else if size < 16 && v.len() == 16 {
                Cow::Borrowed(&v[0..size])
            } else {
                return Err(MirEvalError::InvalidConst(konst.clone()));
            }
        } else {
            Cow::Borrowed(v)
        };
        let addr = self.heap_allocate(size, align)?;
        self.write_memory(addr, &v)?;
        self.patch_addresses(
            &patch_map,
            |bytes| match memory_map {
                MemoryMap::Empty | MemoryMap::Simple(_) => {
                    Err(MirEvalError::InvalidVTableId(from_bytes!(usize, bytes)))
                }
                MemoryMap::Complex(cm) => cm.vtable.ty_of_bytes(bytes),
            },
            addr,
            ty,
            locals,
        )?;
        Ok(Interval::new(addr, size))
    }

    fn eval_place(&mut self, p: &Place, locals: &Locals) -> Result<Interval> {
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
            Stack(it) => (&self.stack, it),
            Heap(it) => (&self.heap, it),
            Invalid(it) => {
                return Err(MirEvalError::UndefinedBehavior(format!(
                    "read invalid memory address {it} with size {size}"
                )));
            }
        };
        mem.get(pos..pos + size)
            .ok_or_else(|| MirEvalError::UndefinedBehavior("out of bound memory read".to_owned()))
    }

    fn write_memory_using_ref(&mut self, addr: Address, size: usize) -> Result<&mut [u8]> {
        let (mem, pos) = match addr {
            Stack(it) => (&mut self.stack, it),
            Heap(it) => (&mut self.heap, it),
            Invalid(it) => {
                return Err(MirEvalError::UndefinedBehavior(format!(
                    "write invalid memory address {it} with size {size}"
                )));
            }
        };
        mem.get_mut(pos..pos + size)
            .ok_or_else(|| MirEvalError::UndefinedBehavior("out of bound memory write".to_owned()))
    }

    fn write_memory(&mut self, addr: Address, r: &[u8]) -> Result<()> {
        if r.is_empty() {
            return Ok(());
        }
        self.write_memory_using_ref(addr, r.len())?.copy_from_slice(r);
        Ok(())
    }

    fn copy_from_interval_or_owned(&mut self, addr: Address, r: IntervalOrOwned) -> Result<()> {
        match r {
            IntervalOrOwned::Borrowed(r) => self.copy_from_interval(addr, r),
            IntervalOrOwned::Owned(r) => self.write_memory(addr, &r),
        }
    }

    fn copy_from_interval(&mut self, addr: Address, r: Interval) -> Result<()> {
        if r.size == 0 {
            return Ok(());
        }

        let oob = || MirEvalError::UndefinedBehavior("out of bounds memory write".to_owned());

        match (addr, r.addr) {
            (Stack(dst), Stack(src)) => {
                if self.stack.len() < src + r.size || self.stack.len() < dst + r.size {
                    return Err(oob());
                }
                self.stack.copy_within(src..src + r.size, dst)
            }
            (Heap(dst), Heap(src)) => {
                if self.stack.len() < src + r.size || self.stack.len() < dst + r.size {
                    return Err(oob());
                }
                self.heap.copy_within(src..src + r.size, dst)
            }
            (Stack(dst), Heap(src)) => {
                self.stack
                    .get_mut(dst..dst + r.size)
                    .ok_or_else(oob)?
                    .copy_from_slice(self.heap.get(src..src + r.size).ok_or_else(oob)?);
            }
            (Heap(dst), Stack(src)) => {
                self.heap
                    .get_mut(dst..dst + r.size)
                    .ok_or_else(oob)?
                    .copy_from_slice(self.stack.get(src..src + r.size).ok_or_else(oob)?);
            }
            _ => {
                return Err(MirEvalError::UndefinedBehavior(format!(
                    "invalid memory write at address {addr:?}"
                )));
            }
        }

        Ok(())
    }

    fn size_align_of(&self, ty: &Ty, locals: &Locals) -> Result<Option<(usize, usize)>> {
        if let Some(layout) = self.layout_cache.borrow().get(ty) {
            return Ok(layout
                .is_sized()
                .then(|| (layout.size.bytes_usize(), layout.align.abi.bytes() as usize)));
        }
        if let DefWithBodyId::VariantId(f) = locals.body.owner {
            if let Some((AdtId::EnumId(e), _)) = ty.as_adt() {
                if f.lookup(self.db).parent == e {
                    // Computing the exact size of enums require resolving the enum discriminants. In order to prevent loops (and
                    // infinite sized type errors) we use a dummy size
                    return Ok(Some((16, 16)));
                }
            }
        }
        let layout = self.layout(ty);
        if self.assert_placeholder_ty_is_unused
            && matches!(layout, Err(MirEvalError::LayoutError(LayoutError::HasPlaceholder, _)))
        {
            return Ok(Some((0, 1)));
        }
        let layout = layout?;
        Ok(layout
            .is_sized()
            .then(|| (layout.size.bytes_usize(), layout.align.abi.bytes() as usize)))
    }

    /// A version of `self.size_of` which returns error if the type is unsized. `what` argument should
    /// be something that complete this: `error: type {ty} was unsized. {what} should be sized`
    fn size_of_sized(&self, ty: &Ty, locals: &Locals, what: &'static str) -> Result<usize> {
        match self.size_align_of(ty, locals)? {
            Some(it) => Ok(it.0),
            None => Err(MirEvalError::TypeIsUnsized(ty.clone(), what)),
        }
    }

    /// A version of `self.size_align_of` which returns error if the type is unsized. `what` argument should
    /// be something that complete this: `error: type {ty} was unsized. {what} should be sized`
    fn size_align_of_sized(
        &self,
        ty: &Ty,
        locals: &Locals,
        what: &'static str,
    ) -> Result<(usize, usize)> {
        match self.size_align_of(ty, locals)? {
            Some(it) => Ok(it),
            None => Err(MirEvalError::TypeIsUnsized(ty.clone(), what)),
        }
    }

    fn heap_allocate(&mut self, size: usize, align: usize) -> Result<Address> {
        if !align.is_power_of_two() || align > 10000 {
            return Err(MirEvalError::UndefinedBehavior(format!("Alignment {align} is invalid")));
        }
        while self.heap.len() % align != 0 {
            self.heap.push(0);
        }
        if size.checked_add(self.heap.len()).is_none_or(|x| x > self.memory_limit) {
            return Err(MirEvalError::Panic(format!("Memory allocation of {size} bytes failed")));
        }
        let pos = self.heap.len();
        self.heap.extend(std::iter::repeat_n(0, size));
        Ok(Address::Heap(pos))
    }

    fn detect_fn_trait(&self, def: FunctionId) -> Option<FnTrait> {
        let def = Some(def);
        if def == self.cached_fn_trait_func {
            Some(FnTrait::Fn)
        } else if def == self.cached_fn_mut_trait_func {
            Some(FnTrait::FnMut)
        } else if def == self.cached_fn_once_trait_func {
            Some(FnTrait::FnOnce)
        } else {
            None
        }
    }

    fn create_memory_map(
        &self,
        bytes: &[u8],
        ty: &Ty,
        locals: &Locals,
    ) -> Result<ComplexMemoryMap> {
        fn rec(
            this: &Evaluator<'_>,
            bytes: &[u8],
            ty: &Ty,
            locals: &Locals,
            mm: &mut ComplexMemoryMap,
            stack_depth_limit: usize,
        ) -> Result<()> {
            if stack_depth_limit.checked_sub(1).is_none() {
                return Err(MirEvalError::StackOverflow);
            }
            match ty.kind(Interner) {
                TyKind::Ref(_, _, t) => {
                    let size = this.size_align_of(t, locals)?;
                    match size {
                        Some((size, _)) => {
                            let addr_usize = from_bytes!(usize, bytes);
                            mm.insert(
                                addr_usize,
                                this.read_memory(Address::from_usize(addr_usize), size)?.into(),
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
                            mm.insert(addr.to_usize(), b.into());
                            if let Some(ty) = check_inner {
                                for i in 0..count {
                                    let offset = element_size * i;
                                    rec(
                                        this,
                                        &b[offset..offset + element_size],
                                        ty,
                                        locals,
                                        mm,
                                        stack_depth_limit - 1,
                                    )?;
                                }
                            }
                        }
                    }
                }
                chalk_ir::TyKind::Array(inner, len) => {
                    let len = match try_const_usize(this.db, len) {
                        Some(it) => it as usize,
                        None => not_supported!("non evaluatable array len in patching addresses"),
                    };
                    let size = this.size_of_sized(inner, locals, "inner of array")?;
                    for i in 0..len {
                        let offset = i * size;
                        rec(
                            this,
                            &bytes[offset..offset + size],
                            inner,
                            locals,
                            mm,
                            stack_depth_limit - 1,
                        )?;
                    }
                }
                chalk_ir::TyKind::Tuple(_, subst) => {
                    let layout = this.layout(ty)?;
                    for (id, ty) in subst.iter(Interner).enumerate() {
                        let ty = ty.assert_ty_ref(Interner); // Tuple only has type argument
                        let offset = layout.fields.offset(id).bytes_usize();
                        let size = this.layout(ty)?.size.bytes_usize();
                        rec(
                            this,
                            &bytes[offset..offset + size],
                            ty,
                            locals,
                            mm,
                            stack_depth_limit - 1,
                        )?;
                    }
                }
                chalk_ir::TyKind::Adt(adt, subst) => match adt.0 {
                    AdtId::StructId(s) => {
                        let data = s.fields(this.db);
                        let layout = this.layout(ty)?;
                        let field_types = this.db.field_types(s.into());
                        for (f, _) in data.fields().iter() {
                            let offset = layout
                                .fields
                                .offset(u32::from(f.into_raw()) as usize)
                                .bytes_usize();
                            let ty = &field_types[f].clone().substitute(Interner, subst);
                            let size = this.layout(ty)?.size.bytes_usize();
                            rec(
                                this,
                                &bytes[offset..offset + size],
                                ty,
                                locals,
                                mm,
                                stack_depth_limit - 1,
                            )?;
                        }
                    }
                    AdtId::EnumId(e) => {
                        let layout = this.layout(ty)?;
                        if let Some((v, l)) = detect_variant_from_bytes(
                            &layout,
                            this.db,
                            &this.target_data_layout,
                            bytes,
                            e,
                        ) {
                            let data = v.fields(this.db);
                            let field_types = this.db.field_types(v.into());
                            for (f, _) in data.fields().iter() {
                                let offset =
                                    l.fields.offset(u32::from(f.into_raw()) as usize).bytes_usize();
                                let ty = &field_types[f].clone().substitute(Interner, subst);
                                let size = this.layout(ty)?.size.bytes_usize();
                                rec(
                                    this,
                                    &bytes[offset..offset + size],
                                    ty,
                                    locals,
                                    mm,
                                    stack_depth_limit - 1,
                                )?;
                            }
                        }
                    }
                    AdtId::UnionId(_) => (),
                },
                _ => (),
            }
            Ok(())
        }
        let mut mm = ComplexMemoryMap::default();
        rec(self, bytes, ty, locals, &mut mm, self.stack_depth_limit - 1)?;
        Ok(mm)
    }

    fn patch_addresses<'vtable>(
        &mut self,
        patch_map: &FxHashMap<usize, usize>,
        ty_of_bytes: impl Fn(&[u8]) -> Result<&'vtable Ty> + Copy,
        addr: Address,
        ty: &Ty,
        locals: &Locals,
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
                        if let Some(it) = patch_map.get(&current) {
                            self.write_memory(addr, &it.to_le_bytes())?;
                        }
                    }
                    None => {
                        let current = from_bytes!(usize, self.read_memory(addr, my_size / 2)?);
                        if let Some(it) = patch_map.get(&current) {
                            self.write_memory(addr, &it.to_le_bytes())?;
                        }
                    }
                }
            }
            TyKind::Function(_) => {
                let ty = ty_of_bytes(self.read_memory(addr, my_size)?)?.clone();
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
                            ty_of_bytes,
                            addr.offset(offset),
                            &ty,
                            locals,
                        )?;
                    }
                }
                AdtId::UnionId(_) => (),
                AdtId::EnumId(e) => {
                    if let Some((ev, layout)) = detect_variant_from_bytes(
                        &layout,
                        self.db,
                        &self.target_data_layout,
                        self.read_memory(addr, layout.size.bytes_usize())?,
                        e,
                    ) {
                        for (i, (_, ty)) in self.db.field_types(ev.into()).iter().enumerate() {
                            let offset = layout.fields.offset(i).bytes_usize();
                            let ty = ty.clone().substitute(Interner, subst);
                            self.patch_addresses(
                                patch_map,
                                ty_of_bytes,
                                addr.offset(offset),
                                &ty,
                                locals,
                            )?;
                        }
                    }
                }
            },
            TyKind::Tuple(_, subst) => {
                for (id, ty) in subst.iter(Interner).enumerate() {
                    let ty = ty.assert_ty_ref(Interner); // Tuple only has type argument
                    let offset = layout.fields.offset(id).bytes_usize();
                    self.patch_addresses(patch_map, ty_of_bytes, addr.offset(offset), ty, locals)?;
                }
            }
            TyKind::Array(inner, len) => {
                let len = match try_const_usize(self.db, len) {
                    Some(it) => it as usize,
                    None => not_supported!("non evaluatable array len in patching addresses"),
                };
                let size = self.size_of_sized(inner, locals, "inner of array")?;
                for i in 0..len {
                    self.patch_addresses(
                        patch_map,
                        ty_of_bytes,
                        addr.offset(i * size),
                        inner,
                        locals,
                    )?;
                }
            }
            TyKind::AssociatedType(_, _)
            | TyKind::Scalar(_)
            | TyKind::Slice(_)
            | TyKind::Raw(_, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Str
            | TyKind::Never
            | TyKind::Closure(_, _)
            | TyKind::Coroutine(_, _)
            | TyKind::CoroutineWitness(_, _)
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
        locals: &Locals,
        target_bb: Option<BasicBlockId>,
        span: MirSpan,
    ) -> Result<Option<StackFrame>> {
        let id = from_bytes!(usize, bytes.get(self)?);
        let next_ty = self.vtable_map.ty(id)?.clone();
        match next_ty.kind(Interner) {
            TyKind::FnDef(def, generic_args) => {
                self.exec_fn_def(*def, generic_args, destination, args, locals, target_bb, span)
            }
            TyKind::Closure(id, subst) => {
                self.exec_closure(*id, bytes.slice(0..0), subst, destination, args, locals, span)
            }
            _ => Err(MirEvalError::InternalError("function pointer to non function".into())),
        }
    }

    fn exec_closure(
        &mut self,
        closure: ClosureId,
        closure_data: Interval,
        generic_args: &Substitution,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals,
        span: MirSpan,
    ) -> Result<Option<StackFrame>> {
        let mir_body = self
            .db
            .monomorphized_mir_body_for_closure(
                closure.into(),
                generic_args.clone(),
                self.trait_env.clone(),
            )
            .map_err(|it| MirEvalError::MirLowerErrorForClosure(closure, it))?;
        let closure_data = if mir_body.locals[mir_body.param_locals[0]].ty.as_reference().is_some()
        {
            closure_data.addr.to_bytes().to_vec()
        } else {
            closure_data.get(self)?.to_owned()
        };
        let arg_bytes = iter::once(Ok(closure_data))
            .chain(args.iter().map(|it| Ok(it.get(self)?.to_owned())))
            .collect::<Result<Vec<_>>>()?;
        let interval = self
            .interpret_mir(mir_body, arg_bytes.into_iter().map(IntervalOrOwned::Owned))
            .map_err(|e| {
                MirEvalError::InFunction(
                    Box::new(e),
                    vec![(Either::Right(closure), span, locals.body.owner)],
                )
            })?;
        destination.write_from_interval(self, interval)?;
        Ok(None)
    }

    fn exec_fn_def(
        &mut self,
        def: FnDefId,
        generic_args: &Substitution,
        destination: Interval,
        args: &[IntervalAndTy],
        locals: &Locals,
        target_bb: Option<BasicBlockId>,
        span: MirSpan,
    ) -> Result<Option<StackFrame>> {
        let def: CallableDefId = from_chalk(self.db, def);
        let generic_args = generic_args.clone();
        match def {
            CallableDefId::FunctionId(def) => {
                if self.detect_fn_trait(def).is_some() {
                    return self.exec_fn_trait(
                        def,
                        args,
                        generic_args,
                        locals,
                        destination,
                        target_bb,
                        span,
                    );
                }
                self.exec_fn_with_args(
                    def,
                    args,
                    generic_args,
                    locals,
                    destination,
                    target_bb,
                    span,
                )
            }
            CallableDefId::StructId(id) => {
                let (size, variant_layout, tag) =
                    self.layout_of_variant(id.into(), generic_args, locals)?;
                let result = self.construct_with_layout(
                    size,
                    &variant_layout,
                    tag,
                    args.iter().map(|it| it.interval.into()),
                )?;
                destination.write_from_bytes(self, &result)?;
                Ok(None)
            }
            CallableDefId::EnumVariantId(id) => {
                let (size, variant_layout, tag) =
                    self.layout_of_variant(id.into(), generic_args, locals)?;
                let result = self.construct_with_layout(
                    size,
                    &variant_layout,
                    tag,
                    args.iter().map(|it| it.interval.into()),
                )?;
                destination.write_from_bytes(self, &result)?;
                Ok(None)
            }
        }
    }

    fn get_mir_or_dyn_index(
        &self,
        def: FunctionId,
        generic_args: Substitution,
        locals: &Locals,
        span: MirSpan,
    ) -> Result<MirOrDynIndex> {
        let pair = (def, generic_args);
        if let Some(r) = self.mir_or_dyn_index_cache.borrow().get(&pair) {
            return Ok(r.clone());
        }
        let (def, generic_args) = pair;
        let r = if let Some(self_ty_idx) =
            is_dyn_method(self.db, self.trait_env.clone(), def, generic_args.clone())
        {
            MirOrDynIndex::Dyn(self_ty_idx)
        } else {
            let (imp, generic_args) =
                self.db.lookup_impl_method(self.trait_env.clone(), def, generic_args.clone());

            let mir_body = self
                .db
                .monomorphized_mir_body(imp.into(), generic_args, self.trait_env.clone())
                .map_err(|e| {
                    MirEvalError::InFunction(
                        Box::new(MirEvalError::MirLowerError(imp, e)),
                        vec![(Either::Left(imp), span, locals.body.owner)],
                    )
                })?;
            MirOrDynIndex::Mir(mir_body)
        };
        self.mir_or_dyn_index_cache.borrow_mut().insert((def, generic_args), r.clone());
        Ok(r)
    }

    fn exec_fn_with_args(
        &mut self,
        mut def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: Substitution,
        locals: &Locals,
        destination: Interval,
        target_bb: Option<BasicBlockId>,
        span: MirSpan,
    ) -> Result<Option<StackFrame>> {
        if self.detect_and_exec_special_function(
            def,
            args,
            &generic_args,
            locals,
            destination,
            span,
        )? {
            return Ok(None);
        }
        if let Some(redirect_def) = self.detect_and_redirect_special_function(def)? {
            def = redirect_def;
        }
        let arg_bytes = args.iter().map(|it| IntervalOrOwned::Borrowed(it.interval));
        match self.get_mir_or_dyn_index(def, generic_args.clone(), locals, span)? {
            MirOrDynIndex::Dyn(self_ty_idx) => {
                // In the layout of current possible receiver, which at the moment of writing this code is one of
                // `&T`, `&mut T`, `Box<T>`, `Rc<T>`, `Arc<T>`, and `Pin<P>` where `P` is one of possible receivers,
                // the vtable is exactly in the `[ptr_size..2*ptr_size]` bytes. So we can use it without branching on
                // the type.
                let first_arg = arg_bytes.clone().next().unwrap();
                let first_arg = first_arg.get(self)?;
                let ty = self
                    .vtable_map
                    .ty_of_bytes(&first_arg[self.ptr_size()..self.ptr_size() * 2])?;
                let mut args_for_target = args.to_vec();
                args_for_target[0] = IntervalAndTy {
                    interval: args_for_target[0].interval.slice(0..self.ptr_size()),
                    ty: ty.clone(),
                };
                let ty = ty.clone().cast(Interner);
                let generics_for_target = Substitution::from_iter(
                    Interner,
                    generic_args
                        .iter(Interner)
                        .enumerate()
                        .map(|(i, it)| if i == self_ty_idx { &ty } else { it }),
                );
                self.exec_fn_with_args(
                    def,
                    &args_for_target,
                    generics_for_target,
                    locals,
                    destination,
                    target_bb,
                    span,
                )
            }
            MirOrDynIndex::Mir(body) => self.exec_looked_up_function(
                body,
                locals,
                def,
                arg_bytes,
                span,
                destination,
                target_bb,
            ),
        }
    }

    fn exec_looked_up_function(
        &mut self,
        mir_body: Arc<MirBody>,
        locals: &Locals,
        def: FunctionId,
        arg_bytes: impl Iterator<Item = IntervalOrOwned>,
        span: MirSpan,
        destination: Interval,
        target_bb: Option<BasicBlockId>,
    ) -> Result<Option<StackFrame>> {
        Ok(if let Some(target_bb) = target_bb {
            let (mut locals, prev_stack_ptr) =
                self.create_locals_for_body(&mir_body, Some(destination))?;
            self.fill_locals_for_body(&mir_body, &mut locals, arg_bytes.into_iter())?;
            let span = (span, locals.body.owner);
            Some(StackFrame { locals, destination: Some(target_bb), prev_stack_ptr, span })
        } else {
            let result = self.interpret_mir(mir_body, arg_bytes).map_err(|e| {
                MirEvalError::InFunction(
                    Box::new(e),
                    vec![(Either::Left(def), span, locals.body.owner)],
                )
            })?;
            destination.write_from_interval(self, result)?;
            None
        })
    }

    fn exec_fn_trait(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: Substitution,
        locals: &Locals,
        destination: Interval,
        target_bb: Option<BasicBlockId>,
        span: MirSpan,
    ) -> Result<Option<StackFrame>> {
        let func = args
            .first()
            .ok_or_else(|| MirEvalError::InternalError("fn trait with no arg".into()))?;
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
        match &func_ty.kind(Interner) {
            TyKind::FnDef(def, subst) => {
                self.exec_fn_def(*def, subst, destination, &args[1..], locals, target_bb, span)
            }
            TyKind::Function(_) => {
                self.exec_fn_pointer(func_data, destination, &args[1..], locals, target_bb, span)
            }
            TyKind::Closure(closure, subst) => self.exec_closure(
                *closure,
                func_data,
                &Substitution::from_iter(Interner, ClosureSubst(subst).parent_subst()),
                destination,
                &args[1..],
                locals,
                span,
            ),
            _ => {
                // try to execute the manual impl of `FnTrait` for structs (nightly feature used in std)
                let arg0 = func;
                let args = &args[1..];
                let arg1 = {
                    let ty = TyKind::Tuple(
                        args.len(),
                        Substitution::from_iter(Interner, args.iter().map(|it| it.ty.clone())),
                    )
                    .intern(Interner);
                    let layout = self.layout(&ty)?;
                    let result = self.construct_with_layout(
                        layout.size.bytes_usize(),
                        &layout,
                        None,
                        args.iter().map(|it| IntervalOrOwned::Borrowed(it.interval)),
                    )?;
                    // FIXME: there is some leak here
                    let size = layout.size.bytes_usize();
                    let addr = self.heap_allocate(size, layout.align.abi.bytes() as usize)?;
                    self.write_memory(addr, &result)?;
                    IntervalAndTy { interval: Interval { addr, size }, ty }
                };
                self.exec_fn_with_args(
                    def,
                    &[arg0.clone(), arg1],
                    generic_args,
                    locals,
                    destination,
                    target_bb,
                    span,
                )
            }
        }
    }

    fn eval_static(&mut self, st: StaticId, locals: &Locals) -> Result<Address> {
        if let Some(o) = self.static_locations.get(&st) {
            return Ok(*o);
        };
        let static_data = self.db.static_signature(st);
        let result = if !static_data.flags.contains(StaticFlags::EXTERN) {
            let konst = self.db.const_eval_static(st).map_err(|e| {
                MirEvalError::ConstEvalError(static_data.name.as_str().to_owned(), Box::new(e))
            })?;
            self.allocate_const_in_heap(locals, &konst)?
        } else {
            let ty = &self.db.infer(st.into())[self.db.body(st.into()).body_expr];
            let Some((size, align)) = self.size_align_of(ty, locals)? else {
                not_supported!("unsized extern static");
            };
            let addr = self.heap_allocate(size, align)?;
            Interval::new(addr, size)
        };
        let addr = self.heap_allocate(self.ptr_size(), self.ptr_size())?;
        self.write_memory(addr, &result.addr.to_bytes())?;
        self.static_locations.insert(st, addr);
        Ok(addr)
    }

    fn const_eval_discriminant(&self, variant: EnumVariantId) -> Result<i128> {
        let r = self.db.const_eval_discriminant(variant);
        match r {
            Ok(r) => Ok(r),
            Err(e) => {
                let db = self.db;
                let loc = variant.lookup(db);
                let edition = self.crate_id.data(self.db).edition;
                let name = format!(
                    "{}::{}",
                    self.db.enum_signature(loc.parent).name.display(db, edition),
                    loc.parent
                        .enum_variants(self.db)
                        .variant_name_by_id(variant)
                        .unwrap()
                        .display(db, edition),
                );
                Err(MirEvalError::ConstEvalError(name, Box::new(e)))
            }
        }
    }

    fn drop_place(&mut self, place: &Place, locals: &mut Locals, span: MirSpan) -> Result<()> {
        let (addr, ty, metadata) = self.place_addr_and_ty_and_metadata(place, locals)?;
        if !locals.drop_flags.remove_place(place, &locals.body.projection_store) {
            return Ok(());
        }
        let metadata = match metadata {
            Some(it) => it.get(self)?.to_vec(),
            None => vec![],
        };
        self.run_drop_glue_deep(ty, locals, addr, &metadata, span)
    }

    fn run_drop_glue_deep(
        &mut self,
        ty: Ty,
        locals: &Locals,
        addr: Address,
        _metadata: &[u8],
        span: MirSpan,
    ) -> Result<()> {
        let Some(drop_fn) = (|| {
            let drop_trait = LangItem::Drop.resolve_trait(self.db, self.crate_id)?;
            drop_trait.trait_items(self.db).method_by_name(&Name::new_symbol_root(sym::drop))
        })() else {
            // in some tests we don't have drop trait in minicore, and
            // we can ignore drop in them.
            return Ok(());
        };

        let generic_args = Substitution::from1(Interner, ty.clone());
        if let Ok(MirOrDynIndex::Mir(body)) =
            self.get_mir_or_dyn_index(drop_fn, generic_args, locals, span)
        {
            self.exec_looked_up_function(
                body,
                locals,
                drop_fn,
                iter::once(IntervalOrOwned::Owned(addr.to_bytes().to_vec())),
                span,
                Interval { addr: Address::Invalid(0), size: 0 },
                None,
            )?;
        }
        match ty.kind(Interner) {
            TyKind::Adt(id, subst) => {
                match id.0 {
                    AdtId::StructId(s) => {
                        let data = self.db.struct_signature(s);
                        if data.flags.contains(StructFlags::IS_MANUALLY_DROP) {
                            return Ok(());
                        }
                        let layout = self.layout_adt(id.0, subst.clone())?;
                        let variant_fields = s.fields(self.db);
                        match variant_fields.shape {
                            FieldsShape::Record | FieldsShape::Tuple => {
                                let field_types = self.db.field_types(s.into());
                                for (field, _) in variant_fields.fields().iter() {
                                    let offset = layout
                                        .fields
                                        .offset(u32::from(field.into_raw()) as usize)
                                        .bytes_usize();
                                    let addr = addr.offset(offset);
                                    let ty = field_types[field].clone().substitute(Interner, subst);
                                    self.run_drop_glue_deep(ty, locals, addr, &[], span)?;
                                }
                            }
                            FieldsShape::Unit => (),
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
            | TyKind::Coroutine(_, _)
            | TyKind::CoroutineWitness(_, _)
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

pub fn render_const_using_debug_impl(
    db: &dyn HirDatabase,
    owner: DefWithBodyId,
    c: &Const,
) -> Result<String> {
    let mut evaluator = Evaluator::new(db, owner, false, None)?;
    let locals = &Locals {
        ptr: ArenaMap::new(),
        body: db
            .mir_body(owner)
            .map_err(|_| MirEvalError::NotSupported("unreachable".to_owned()))?,
        drop_flags: DropFlags::default(),
    };
    let data = evaluator.allocate_const_in_heap(locals, c)?;
    let resolver = owner.resolver(db);
    let Some(TypeNs::TraitId(debug_trait)) = resolver.resolve_path_in_type_ns_fully(
        db,
        &hir_def::expr_store::path::Path::from_known_path_with_no_generic(path![core::fmt::Debug]),
    ) else {
        not_supported!("core::fmt::Debug not found");
    };
    let Some(debug_fmt_fn) =
        debug_trait.trait_items(db).method_by_name(&Name::new_symbol_root(sym::fmt))
    else {
        not_supported!("core::fmt::Debug::fmt not found");
    };
    // a1 = &[""]
    let a1 = evaluator.heap_allocate(evaluator.ptr_size() * 2, evaluator.ptr_size())?;
    // a2 = &[::core::fmt::ArgumentV1::new(&(THE_CONST), ::core::fmt::Debug::fmt)]
    // FIXME: we should call the said function, but since its name is going to break in the next rustc version
    // and its ABI doesn't break yet, we put it in memory manually.
    let a2 = evaluator.heap_allocate(evaluator.ptr_size() * 2, evaluator.ptr_size())?;
    evaluator.write_memory(a2, &data.addr.to_bytes())?;
    let debug_fmt_fn_ptr = evaluator.vtable_map.id(TyKind::FnDef(
        CallableDefId::FunctionId(debug_fmt_fn).to_chalk(db),
        Substitution::from1(Interner, c.data(Interner).ty.clone()),
    )
    .intern(Interner));
    evaluator.write_memory(a2.offset(evaluator.ptr_size()), &debug_fmt_fn_ptr.to_le_bytes())?;
    // a3 = ::core::fmt::Arguments::new_v1(a1, a2)
    // FIXME: similarly, we should call function here, not directly working with memory.
    let a3 = evaluator.heap_allocate(evaluator.ptr_size() * 6, evaluator.ptr_size())?;
    evaluator.write_memory(a3, &a1.to_bytes())?;
    evaluator.write_memory(a3.offset(evaluator.ptr_size()), &[1])?;
    evaluator.write_memory(a3.offset(2 * evaluator.ptr_size()), &a2.to_bytes())?;
    evaluator.write_memory(a3.offset(3 * evaluator.ptr_size()), &[1])?;
    let Some(ValueNs::FunctionId(format_fn)) = resolver.resolve_path_in_value_ns_fully(
        db,
        &hir_def::expr_store::path::Path::from_known_path_with_no_generic(path![std::fmt::format]),
        HygieneId::ROOT,
    ) else {
        not_supported!("std::fmt::format not found");
    };
    let interval = evaluator.interpret_mir(
        db.mir_body(format_fn.into()).map_err(|e| MirEvalError::MirLowerError(format_fn, e))?,
        [IntervalOrOwned::Borrowed(Interval { addr: a3, size: evaluator.ptr_size() * 6 })]
            .into_iter(),
    )?;
    let message_string = interval.get(&evaluator)?;
    let addr =
        Address::from_bytes(&message_string[evaluator.ptr_size()..2 * evaluator.ptr_size()])?;
    let size = from_bytes!(usize, message_string[2 * evaluator.ptr_size()..]);
    Ok(std::string::String::from_utf8_lossy(evaluator.read_memory(addr, size)?).into_owned())
}

pub fn pad16(it: &[u8], is_signed: bool) -> [u8; 16] {
    let is_negative = is_signed && it.last().unwrap_or(&0) > &127;
    let mut res = [if is_negative { 255 } else { 0 }; 16];
    res[..it.len()].copy_from_slice(it);
    res
}

macro_rules! for_each_int_type {
    ($call_macro:path, $args:tt) => {
        $call_macro! {
            $args
            I8
            U8
            I16
            U16
            I32
            U32
            I64
            U64
            I128
            U128
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum IntValue {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    I128(i128),
    U128(u128),
}

macro_rules! checked_int_op {
    ( [ $op:ident ] $( $int_ty:ident )+ ) => {
        fn $op(self, other: Self) -> Option<Self> {
            match (self, other) {
                $( (Self::$int_ty(a), Self::$int_ty(b)) => a.$op(b).map(Self::$int_ty), )+
                _ => panic!("incompatible integer types"),
            }
        }
    };
}

macro_rules! int_bit_shifts {
    ( [ $op:ident ] $( $int_ty:ident )+ ) => {
        fn $op(self, amount: u32) -> Option<Self> {
            match self {
                $( Self::$int_ty(this) => this.$op(amount).map(Self::$int_ty), )+
            }
        }
    };
}

macro_rules! unchecked_int_op {
    ( [ $name:ident, $op:tt ]  $( $int_ty:ident )+ ) => {
        fn $name(self, other: Self) -> Self {
            match (self, other) {
                $( (Self::$int_ty(a), Self::$int_ty(b)) => Self::$int_ty(a $op b), )+
                _ => panic!("incompatible integer types"),
            }
        }
    };
}

impl IntValue {
    fn from_bytes(bytes: &[u8], is_signed: bool) -> Self {
        match (bytes.len(), is_signed) {
            (1, false) => Self::U8(u8::from_le_bytes(bytes.try_into().unwrap())),
            (1, true) => Self::I8(i8::from_le_bytes(bytes.try_into().unwrap())),
            (2, false) => Self::U16(u16::from_le_bytes(bytes.try_into().unwrap())),
            (2, true) => Self::I16(i16::from_le_bytes(bytes.try_into().unwrap())),
            (4, false) => Self::U32(u32::from_le_bytes(bytes.try_into().unwrap())),
            (4, true) => Self::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
            (8, false) => Self::U64(u64::from_le_bytes(bytes.try_into().unwrap())),
            (8, true) => Self::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
            (16, false) => Self::U128(u128::from_le_bytes(bytes.try_into().unwrap())),
            (16, true) => Self::I128(i128::from_le_bytes(bytes.try_into().unwrap())),
            (len, is_signed) => {
                never!("invalid integer size: {len}, signed: {is_signed}");
                Self::I32(0)
            }
        }
    }

    fn to_bytes(self) -> Vec<u8> {
        macro_rules! m {
            ( [] $( $int_ty:ident )+ ) => {
                match self {
                    $( Self::$int_ty(v) => v.to_le_bytes().to_vec() ),+
                }
            };
        }
        for_each_int_type! { m, [] }
    }

    fn as_u32(self) -> Option<u32> {
        macro_rules! m {
            ( [] $( $int_ty:ident )+ ) => {
                match self {
                    $( Self::$int_ty(v) => v.try_into().ok() ),+
                }
            };
        }
        for_each_int_type! { m, [] }
    }

    for_each_int_type!(checked_int_op, [checked_add]);
    for_each_int_type!(checked_int_op, [checked_sub]);
    for_each_int_type!(checked_int_op, [checked_div]);
    for_each_int_type!(checked_int_op, [checked_rem]);
    for_each_int_type!(checked_int_op, [checked_mul]);

    for_each_int_type!(int_bit_shifts, [checked_shl]);
    for_each_int_type!(int_bit_shifts, [checked_shr]);
}

impl std::ops::BitAnd for IntValue {
    type Output = Self;
    for_each_int_type!(unchecked_int_op, [bitand, &]);
}
impl std::ops::BitOr for IntValue {
    type Output = Self;
    for_each_int_type!(unchecked_int_op, [bitor, |]);
}
impl std::ops::BitXor for IntValue {
    type Output = Self;
    for_each_int_type!(unchecked_int_op, [bitxor, ^]);
}
