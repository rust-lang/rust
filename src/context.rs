use std::cell::{Cell, RefCell};

use gccjit::{
    Block,
    Context,
    CType,
    Function,
    FunctionType,
    LValue,
    RValue,
    Struct,
    Type,
};
use rustc_codegen_ssa::base::wants_msvc_seh;
use rustc_codegen_ssa::traits::{
    BackendTypes,
    MiscMethods,
};
use rustc_data_structures::base_n;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::span_bug;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::{self, Instance, ParamEnv, PolyExistentialTraitRef, Ty, TyCtxt};
use rustc_middle::ty::layout::{FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasParamEnv, HasTyCtxt, LayoutError, TyAndLayout, LayoutOfHelpers};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use rustc_target::abi::{call::FnAbi, HasDataLayout, PointeeInfo, Size, TargetDataLayout, VariantIdx};
use rustc_target::spec::{HasTargetSpec, Target, TlsModel};

use crate::callee::get_fn;
use crate::declare::mangle_name;

#[derive(Clone)]
pub struct FuncSig<'gcc> {
    pub params: Vec<Type<'gcc>>,
    pub return_type: Type<'gcc>,
}

pub struct CodegenCx<'gcc, 'tcx> {
    pub check_overflow: bool,
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,
    pub context: &'gcc Context<'gcc>,

    // TODO(antoyo): First set it to a dummy block to avoid using Option?
    pub current_block: RefCell<Option<Block<'gcc>>>,
    pub current_func: RefCell<Option<Function<'gcc>>>,
    pub normal_function_addresses: RefCell<FxHashSet<RValue<'gcc>>>,

    pub functions: RefCell<FxHashMap<String, Function<'gcc>>>,

    pub tls_model: gccjit::TlsModel,

    pub bool_type: Type<'gcc>,
    pub i8_type: Type<'gcc>,
    pub i16_type: Type<'gcc>,
    pub i32_type: Type<'gcc>,
    pub i64_type: Type<'gcc>,
    pub i128_type: Type<'gcc>,
    pub isize_type: Type<'gcc>,

    pub u8_type: Type<'gcc>,
    pub u16_type: Type<'gcc>,
    pub u32_type: Type<'gcc>,
    pub u64_type: Type<'gcc>,
    pub u128_type: Type<'gcc>,
    pub usize_type: Type<'gcc>,

    pub int_type: Type<'gcc>,
    pub uint_type: Type<'gcc>,
    pub long_type: Type<'gcc>,
    pub ulong_type: Type<'gcc>,
    pub ulonglong_type: Type<'gcc>,
    pub sizet_type: Type<'gcc>,

    pub float_type: Type<'gcc>,
    pub double_type: Type<'gcc>,

    pub linkage: Cell<FunctionType>,
    pub scalar_types: RefCell<FxHashMap<Ty<'tcx>, Type<'gcc>>>,
    pub types: RefCell<FxHashMap<(Ty<'tcx>, Option<VariantIdx>), Type<'gcc>>>,
    pub tcx: TyCtxt<'tcx>,

    pub struct_types: RefCell<FxHashMap<Vec<Type<'gcc>>, Type<'gcc>>>,

    pub types_with_fields_to_set: RefCell<FxHashMap<Type<'gcc>, (Struct<'gcc>, TyAndLayout<'tcx>)>>,

    /// Cache instances of monomorphic and polymorphic items
    pub instances: RefCell<FxHashMap<Instance<'tcx>, LValue<'gcc>>>,
    /// Cache function instances of monomorphic and polymorphic items
    pub function_instances: RefCell<FxHashMap<Instance<'tcx>, RValue<'gcc>>>,
    /// Cache generated vtables
    pub vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), RValue<'gcc>>>,

    /// Cache of emitted const globals (value -> global)
    pub const_globals: RefCell<FxHashMap<RValue<'gcc>, RValue<'gcc>>>,

    /// Cache of constant strings,
    pub const_cstr_cache: RefCell<FxHashMap<Symbol, LValue<'gcc>>>,

    /// Cache of globals.
    pub globals: RefCell<FxHashMap<String, RValue<'gcc>>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,
    pub global_gen_sym_counter: Cell<usize>,

    eh_personality: Cell<Option<RValue<'gcc>>>,

    pub pointee_infos: RefCell<FxHashMap<(Ty<'tcx>, Size), Option<PointeeInfo>>>,

    /// NOTE: a hack is used because the rustc API is not suitable to libgccjit and as such,
    /// `const_undef()` returns struct as pointer so that they can later be assigned a value.
    /// As such, this set remembers which of these pointers were returned by this function so that
    /// they can be deferenced later.
    /// FIXME(antoyo): fix the rustc API to avoid having this hack.
    pub structs_as_pointer: RefCell<FxHashSet<RValue<'gcc>>>,
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn new(context: &'gcc Context<'gcc>, codegen_unit: &'tcx CodegenUnit<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let check_overflow = tcx.sess.overflow_checks();
        // TODO(antoyo): fix this mess. libgccjit seems to return random type when using new_int_type().
        let isize_type = context.new_c_type(CType::LongLong);
        let usize_type = context.new_c_type(CType::ULongLong);
        let bool_type = context.new_type::<bool>();
        let i8_type = context.new_type::<i8>();
        let i16_type = context.new_type::<i16>();
        let i32_type = context.new_type::<i32>();
        let i64_type = context.new_c_type(CType::LongLong);
        let i128_type = context.new_c_type(CType::Int128t).get_aligned(8); // TODO(antoyo): should the alignment be hard-coded?
        let u8_type = context.new_type::<u8>();
        let u16_type = context.new_type::<u16>();
        let u32_type = context.new_type::<u32>();
        let u64_type = context.new_c_type(CType::ULongLong);
        let u128_type = context.new_c_type(CType::UInt128t).get_aligned(8); // TODO(antoyo): should the alignment be hard-coded?

        let tls_model = to_gcc_tls_mode(tcx.sess.tls_model());

        let float_type = context.new_type::<f32>();
        let double_type = context.new_type::<f64>();

        let int_type = context.new_c_type(CType::Int);
        let uint_type = context.new_c_type(CType::UInt);
        let long_type = context.new_c_type(CType::Long);
        let ulong_type = context.new_c_type(CType::ULong);
        let ulonglong_type = context.new_c_type(CType::ULongLong);
        let sizet_type = context.new_c_type(CType::SizeT);

        assert_eq!(isize_type, i64_type);
        assert_eq!(usize_type, u64_type);

        let mut functions = FxHashMap::default();
        let builtins = [
            "__builtin_unreachable", "abort", "__builtin_expect", "__builtin_add_overflow", "__builtin_mul_overflow",
            "__builtin_saddll_overflow", /*"__builtin_sadd_overflow",*/ "__builtin_smulll_overflow", /*"__builtin_smul_overflow",*/
            "__builtin_ssubll_overflow", /*"__builtin_ssub_overflow",*/ "__builtin_sub_overflow", "__builtin_uaddll_overflow",
            "__builtin_uadd_overflow", "__builtin_umulll_overflow", "__builtin_umul_overflow", "__builtin_usubll_overflow",
            "__builtin_usub_overflow", "sqrtf", "sqrt", "__builtin_powif", "__builtin_powi", "sinf", "sin", "cosf", "cos",
            "powf", "pow", "expf", "exp", "exp2f", "exp2", "logf", "log", "log10f", "log10", "log2f", "log2", "fmaf",
            "fma", "fabsf", "fabs", "fminf", "fmin", "fmaxf", "fmax", "copysignf", "copysign", "floorf", "floor", "ceilf",
            "ceil", "truncf", "trunc", "rintf", "rint", "nearbyintf", "nearbyint", "roundf", "round",
            "__builtin_expect_with_probability",
        ];

        for builtin in builtins.iter() {
            functions.insert(builtin.to_string(), context.get_builtin_function(builtin));
        }

        Self {
            check_overflow,
            codegen_unit,
            context,
            current_block: RefCell::new(None),
            current_func: RefCell::new(None),
            normal_function_addresses: Default::default(),
            functions: RefCell::new(functions),

            tls_model,

            bool_type,
            i8_type,
            i16_type,
            i32_type,
            i64_type,
            i128_type,
            isize_type,
            usize_type,
            u8_type,
            u16_type,
            u32_type,
            u64_type,
            u128_type,
            int_type,
            uint_type,
            long_type,
            ulong_type,
            ulonglong_type,
            sizet_type,

            float_type,
            double_type,

            linkage: Cell::new(FunctionType::Internal),
            instances: Default::default(),
            function_instances: Default::default(),
            vtables: Default::default(),
            const_globals: Default::default(),
            const_cstr_cache: Default::default(),
            globals: Default::default(),
            scalar_types: Default::default(),
            types: Default::default(),
            tcx,
            struct_types: Default::default(),
            types_with_fields_to_set: Default::default(),
            local_gen_sym_counter: Cell::new(0),
            global_gen_sym_counter: Cell::new(0),
            eh_personality: Cell::new(None),
            pointee_infos: Default::default(),
            structs_as_pointer: Default::default(),
        }
    }

    pub fn rvalue_as_function(&self, value: RValue<'gcc>) -> Function<'gcc> {
        let function: Function<'gcc> = unsafe { std::mem::transmute(value) };
        debug_assert!(self.functions.borrow().values().find(|value| **value == function).is_some(),
            "{:?} ({:?}) is not a function", value, value.get_type());
        function
    }

    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }
}

impl<'gcc, 'tcx> BackendTypes for CodegenCx<'gcc, 'tcx> {
    type Value = RValue<'gcc>;
    type Function = RValue<'gcc>;

    type BasicBlock = Block<'gcc>;
    type Type = Type<'gcc>;
    type Funclet = (); // TODO(antoyo)

    type DIScope = (); // TODO(antoyo)
    type DILocation = (); // TODO(antoyo)
    type DIVariable = (); // TODO(antoyo)
}

impl<'gcc, 'tcx> MiscMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn vtables(&self) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<PolyExistentialTraitRef<'tcx>>), RValue<'gcc>>> {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> RValue<'gcc> {
        let func = get_fn(self, instance);
        *self.current_func.borrow_mut() = Some(self.rvalue_as_function(func));
        func
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> RValue<'gcc> {
        let func = get_fn(self, instance);
        let func = self.rvalue_as_function(func);
        let ptr = func.get_address(None);

        // TODO(antoyo): don't do this twice: i.e. in declare_fn and here.
        // FIXME(antoyo): the rustc API seems to call get_fn_addr() when not needed (e.g. for FFI).

        self.normal_function_addresses.borrow_mut().insert(ptr);

        ptr
    }

    fn eh_personality(&self) -> RValue<'gcc> {
        // The exception handling personality function.
        //
        // If our compilation unit has the `eh_personality` lang item somewhere
        // within it, then we just need to codegen that. Otherwise, we're
        // building an rlib which will depend on some upstream implementation of
        // this function, so we just codegen a generic reference to it. We don't
        // specify any of the types for the function, we just make it a symbol
        // that LLVM can later use.
        //
        // Note that MSVC is a little special here in that we don't use the
        // `eh_personality` lang item at all. Currently LLVM has support for
        // both Dwarf and SEH unwind mechanisms for MSVC targets and uses the
        // *name of the personality function* to decide what kind of unwind side
        // tables/landing pads to emit. It looks like Dwarf is used by default,
        // injecting a dependency on the `_Unwind_Resume` symbol for resuming
        // an "exception", but for MSVC we want to force SEH. This means that we
        // can't actually have the personality function be our standard
        // `rust_eh_personality` function, but rather we wired it up to the
        // CRT's custom personality function, which forces LLVM to consider
        // landing pads as "landing pads for SEH".
        if let Some(llpersonality) = self.eh_personality.get() {
            return llpersonality;
        }
        let tcx = self.tcx;
        let llfn = match tcx.lang_items().eh_personality() {
            Some(def_id) if !wants_msvc_seh(self.sess()) => self.get_fn_addr(
                ty::Instance::resolve(
                    tcx,
                    ty::ParamEnv::reveal_all(),
                    def_id,
                    tcx.intern_substs(&[]),
                )
                .unwrap().unwrap(),
            ),
            _ => {
                let _name = if wants_msvc_seh(self.sess()) {
                    "__CxxFrameHandler3"
                } else {
                    "rust_eh_personality"
                };
                //let func = self.declare_func(name, self.type_i32(), &[], true);
                // FIXME(antoyo): this hack should not be needed. That will probably be removed when
                // unwinding support is added.
                self.context.new_rvalue_from_int(self.int_type, 0)
            }
        };
        // TODO(antoyo): apply target cpu attributes.
        self.eh_personality.set(Some(llfn));
        llfn
    }

    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn check_overflow(&self) -> bool {
        self.check_overflow
    }

    fn codegen_unit(&self) -> &'tcx CodegenUnit<'tcx> {
        self.codegen_unit
    }

    fn used_statics(&self) -> &RefCell<Vec<RValue<'gcc>>> {
        unimplemented!();
    }

    fn set_frame_pointer_type(&self, _llfn: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn apply_target_cpu_attr(&self, _llfn: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn create_used_variable(&self) {
        unimplemented!();
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        if self.get_declared_value("main").is_none() {
            Some(self.declare_cfn("main", fn_type))
        }
        else {
            // If the symbol already exists, it is an error: for example, the user wrote
            // #[no_mangle] extern "C" fn main(..) {..}
            // instead of #[start]
            None
        }
    }

    fn compiler_used_statics(&self) -> &RefCell<Vec<RValue<'gcc>>> {
        unimplemented!()
    }

    fn create_compiler_used_variable(&self) {
        unimplemented!()
    }
}

impl<'gcc, 'tcx> HasTyCtxt<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'gcc, 'tcx> HasDataLayout for CodegenCx<'gcc, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'gcc, 'tcx> HasTargetSpec for CodegenCx<'gcc, 'tcx> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target
    }
}

impl<'gcc, 'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'gcc, 'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_) = err {
            self.sess().span_fatal(span, &err.to_string())
        } else {
            span_bug!(span, "failed to get layout for `{}`: {}", ty, err)
        }
    }
}

impl<'gcc, 'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'gcc, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_)) = err {
            self.sess().span_fatal(span, &err.to_string())
        } else {
            match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_fn_ptr({}, {:?})` failed: {}",
                        sig,
                        extra_args,
                        err
                    );
                }
                FnAbiRequest::OfInstance { instance, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({}, {:?})` failed: {}",
                        instance,
                        extra_args,
                        err
                    );
                }
            }
        }
    }
}

impl<'tcx, 'gcc> HasParamEnv<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::reveal_all()
    }
}

impl<'b, 'tcx> CodegenCx<'b, 'tcx> {
    /// Generates a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push_str(".");
        base_n::push_str(idx as u128, base_n::ALPHANUMERIC_ONLY, &mut name);
        name
    }
}

pub fn unit_name<'tcx>(codegen_unit: &CodegenUnit<'tcx>) -> String {
    let name = &codegen_unit.name().to_string();
    mangle_name(&name.replace('-', "_"))
}

fn to_gcc_tls_mode(tls_model: TlsModel) -> gccjit::TlsModel {
    match tls_model {
        TlsModel::GeneralDynamic => gccjit::TlsModel::GlobalDynamic,
        TlsModel::LocalDynamic => gccjit::TlsModel::LocalDynamic,
        TlsModel::InitialExec => gccjit::TlsModel::InitialExec,
        TlsModel::LocalExec => gccjit::TlsModel::LocalExec,
    }
}
