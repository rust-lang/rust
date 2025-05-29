use std::cell::{Cell, RefCell};

use gccjit::{
    Block, CType, Context, Function, FunctionPtrType, FunctionType, LValue, Location, RValue, Type,
};
use rustc_abi::{HasDataLayout, PointeeInfo, Size, TargetDataLayout, VariantIdx};
use rustc_codegen_ssa::base::wants_msvc_seh;
use rustc_codegen_ssa::errors as ssa_errors;
use rustc_codegen_ssa::traits::{BackendTypes, BaseTypeCodegenMethods, MiscCodegenMethods};
use rustc_data_structures::base_n::{ALPHANUMERIC_ONLY, ToBaseN};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOf, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError,
    LayoutOfHelpers,
};
use rustc_middle::ty::{self, ExistentialTraitRef, Instance, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::source_map::respan;
use rustc_span::{DUMMY_SP, Span};
use rustc_target::spec::{
    HasTargetSpec, HasWasmCAbiOpt, HasX86AbiOpt, Target, TlsModel, WasmCAbi, X86Abi,
};

#[cfg(feature = "master")]
use crate::abi::conv_to_fn_attribute;
use crate::callee::get_fn;
use crate::common::SignType;

#[cfg_attr(not(feature = "master"), allow(dead_code))]
pub struct CodegenCx<'gcc, 'tcx> {
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,
    pub context: &'gcc Context<'gcc>,

    // TODO(bjorn3): Can this field be removed?
    pub current_func: RefCell<Option<Function<'gcc>>>,
    pub normal_function_addresses: RefCell<FxHashSet<RValue<'gcc>>>,
    pub function_address_names: RefCell<FxHashMap<RValue<'gcc>, String>>,

    pub functions: RefCell<FxHashMap<String, Function<'gcc>>>,
    pub intrinsics: RefCell<FxHashMap<String, Function<'gcc>>>,

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

    pub char_type: Type<'gcc>,
    pub uchar_type: Type<'gcc>,
    pub short_type: Type<'gcc>,
    pub ushort_type: Type<'gcc>,
    pub int_type: Type<'gcc>,
    pub uint_type: Type<'gcc>,
    pub long_type: Type<'gcc>,
    pub ulong_type: Type<'gcc>,
    pub longlong_type: Type<'gcc>,
    pub ulonglong_type: Type<'gcc>,
    pub sizet_type: Type<'gcc>,

    pub supports_128bit_integers: bool,
    pub supports_f16_type: bool,
    pub supports_f32_type: bool,
    pub supports_f64_type: bool,
    pub supports_f128_type: bool,

    pub float_type: Type<'gcc>,
    pub double_type: Type<'gcc>,

    pub linkage: Cell<FunctionType>,
    pub scalar_types: RefCell<FxHashMap<Ty<'tcx>, Type<'gcc>>>,
    pub types: RefCell<FxHashMap<(Ty<'tcx>, Option<VariantIdx>), Type<'gcc>>>,
    pub tcx: TyCtxt<'tcx>,

    pub struct_types: RefCell<FxHashMap<Vec<Type<'gcc>>, Type<'gcc>>>,

    /// Cache instances of monomorphic and polymorphic items
    pub instances: RefCell<FxHashMap<Instance<'tcx>, LValue<'gcc>>>,
    /// Cache function instances of monomorphic and polymorphic items
    pub function_instances: RefCell<FxHashMap<Instance<'tcx>, Function<'gcc>>>,
    /// Cache generated vtables
    pub vtables:
        RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), RValue<'gcc>>>,

    // TODO(antoyo): improve the SSA API to not require those.
    /// Mapping from function pointer type to indexes of on stack parameters.
    pub on_stack_params: RefCell<FxHashMap<FunctionPtrType<'gcc>, FxHashSet<usize>>>,
    /// Mapping from function to indexes of on stack parameters.
    pub on_stack_function_params: RefCell<FxHashMap<Function<'gcc>, FxHashSet<usize>>>,

    /// Cache of emitted const globals (value -> global)
    pub const_globals: RefCell<FxHashMap<RValue<'gcc>, RValue<'gcc>>>,

    /// Map from the address of a global variable (rvalue) to the global variable itself (lvalue).
    /// TODO(antoyo): remove when the rustc API is fixed.
    pub global_lvalues: RefCell<FxHashMap<RValue<'gcc>, LValue<'gcc>>>,

    /// Cache of constant strings,
    pub const_str_cache: RefCell<FxHashMap<String, LValue<'gcc>>>,

    /// Cache of globals.
    pub globals: RefCell<FxHashMap<String, RValue<'gcc>>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,

    eh_personality: Cell<Option<RValue<'gcc>>>,
    #[cfg(feature = "master")]
    pub rust_try_fn: Cell<Option<(Type<'gcc>, Function<'gcc>)>>,

    pub pointee_infos: RefCell<FxHashMap<(Ty<'tcx>, Size), Option<PointeeInfo>>>,

    /// NOTE: a hack is used because the rustc API is not suitable to libgccjit and as such,
    /// `const_undef()` returns struct as pointer so that they can later be assigned a value.
    /// As such, this set remembers which of these pointers were returned by this function so that
    /// they can be dereferenced later.
    /// FIXME(antoyo): fix the rustc API to avoid having this hack.
    pub structs_as_pointer: RefCell<FxHashSet<RValue<'gcc>>>,

    #[cfg(feature = "master")]
    pub cleanup_blocks: RefCell<FxHashSet<Block<'gcc>>>,
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &'gcc Context<'gcc>,
        codegen_unit: &'tcx CodegenUnit<'tcx>,
        tcx: TyCtxt<'tcx>,
        supports_128bit_integers: bool,
        supports_f16_type: bool,
        supports_f32_type: bool,
        supports_f64_type: bool,
        supports_f128_type: bool,
    ) -> Self {
        let create_type = |ctype, rust_type| {
            let layout = tcx
                .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(rust_type))
                .unwrap();
            let align = layout.align.abi.bytes();
            #[cfg(feature = "master")]
            {
                context.new_c_type(ctype).get_aligned(align)
            }
            #[cfg(not(feature = "master"))]
            {
                // Since libgccjit 12 doesn't contain the fix to compare aligned integer types,
                // only align u128 and i128.
                if layout.ty.int_size_and_signed(tcx).0.bytes() == 16 {
                    context.new_c_type(ctype).get_aligned(align)
                } else {
                    context.new_c_type(ctype)
                }
            }
        };

        let i8_type = create_type(CType::Int8t, tcx.types.i8);
        let i16_type = create_type(CType::Int16t, tcx.types.i16);
        let i32_type = create_type(CType::Int32t, tcx.types.i32);
        let i64_type = create_type(CType::Int64t, tcx.types.i64);
        let u8_type = create_type(CType::UInt8t, tcx.types.u8);
        let u16_type = create_type(CType::UInt16t, tcx.types.u16);
        let u32_type = create_type(CType::UInt32t, tcx.types.u32);
        let u64_type = create_type(CType::UInt64t, tcx.types.u64);

        let (i128_type, u128_type) = if supports_128bit_integers {
            let i128_type = create_type(CType::Int128t, tcx.types.i128);
            let u128_type = create_type(CType::UInt128t, tcx.types.u128);
            (i128_type, u128_type)
        } else {
            /*let layout = tcx.layout_of(ParamEnv::reveal_all().and(tcx.types.i128)).unwrap();
            let i128_align = layout.align.abi.bytes();
            let layout = tcx.layout_of(ParamEnv::reveal_all().and(tcx.types.u128)).unwrap();
            let u128_align = layout.align.abi.bytes();*/

            // TODO(antoyo): re-enable the alignment when libgccjit fixed the issue in
            // gcc_jit_context_new_array_constructor (it should not use reinterpret_cast).
            let i128_type = context.new_array_type(None, i64_type, 2)/*.get_aligned(i128_align)*/;
            let u128_type = context.new_array_type(None, u64_type, 2)/*.get_aligned(u128_align)*/;
            (i128_type, u128_type)
        };

        let tls_model = to_gcc_tls_mode(tcx.sess.tls_model());

        // TODO(antoyo): set alignment on those types as well.
        let float_type = context.new_type::<f32>();
        let double_type = context.new_type::<f64>();

        let char_type = context.new_c_type(CType::Char);
        let uchar_type = context.new_c_type(CType::UChar);
        let short_type = context.new_c_type(CType::Short);
        let ushort_type = context.new_c_type(CType::UShort);
        let int_type = context.new_c_type(CType::Int);
        let uint_type = context.new_c_type(CType::UInt);
        let long_type = context.new_c_type(CType::Long);
        let ulong_type = context.new_c_type(CType::ULong);
        let longlong_type = context.new_c_type(CType::LongLong);
        let ulonglong_type = context.new_c_type(CType::ULongLong);
        let sizet_type = context.new_c_type(CType::SizeT);

        let usize_type = sizet_type;
        let isize_type = usize_type;
        let bool_type = context.new_type::<bool>();

        let mut functions = FxHashMap::default();
        let builtins = ["abort"];

        for builtin in builtins.iter() {
            functions.insert(builtin.to_string(), context.get_builtin_function(builtin));
        }

        let mut cx = Self {
            codegen_unit,
            context,
            current_func: RefCell::new(None),
            normal_function_addresses: Default::default(),
            function_address_names: Default::default(),
            functions: RefCell::new(functions),
            intrinsics: RefCell::new(FxHashMap::default()),

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
            char_type,
            uchar_type,
            short_type,
            ushort_type,
            int_type,
            uint_type,
            long_type,
            ulong_type,
            longlong_type,
            ulonglong_type,
            sizet_type,

            supports_128bit_integers,
            supports_f16_type,
            supports_f32_type,
            supports_f64_type,
            supports_f128_type,

            float_type,
            double_type,

            linkage: Cell::new(FunctionType::Internal),
            instances: Default::default(),
            function_instances: Default::default(),
            on_stack_params: Default::default(),
            on_stack_function_params: Default::default(),
            vtables: Default::default(),
            const_globals: Default::default(),
            global_lvalues: Default::default(),
            const_str_cache: Default::default(),
            globals: Default::default(),
            scalar_types: Default::default(),
            types: Default::default(),
            tcx,
            struct_types: Default::default(),
            local_gen_sym_counter: Cell::new(0),
            eh_personality: Cell::new(None),
            #[cfg(feature = "master")]
            rust_try_fn: Cell::new(None),
            pointee_infos: Default::default(),
            structs_as_pointer: Default::default(),
            #[cfg(feature = "master")]
            cleanup_blocks: Default::default(),
        };
        // TODO(antoyo): instead of doing this, add SsizeT to libgccjit.
        cx.isize_type = usize_type.to_signed(&cx);
        cx
    }

    pub fn rvalue_as_function(&self, value: RValue<'gcc>) -> Function<'gcc> {
        let function: Function<'gcc> = unsafe { std::mem::transmute(value) };
        debug_assert!(
            self.functions.borrow().values().any(|value| *value == function),
            "{:?} ({:?}) is not a function",
            value,
            value.get_type()
        );
        function
    }

    pub fn is_native_int_type(&self, typ: Type<'gcc>) -> bool {
        let types = [
            self.u8_type,
            self.u16_type,
            self.u32_type,
            self.u64_type,
            self.i8_type,
            self.i16_type,
            self.i32_type,
            self.i64_type,
        ];

        for native_type in types {
            if native_type.is_compatible_with(typ) {
                return true;
            }
        }

        self.supports_128bit_integers
            && (self.u128_type.is_compatible_with(typ) || self.i128_type.is_compatible_with(typ))
    }

    pub fn is_non_native_int_type(&self, typ: Type<'gcc>) -> bool {
        !self.supports_128bit_integers
            && (self.u128_type.is_compatible_with(typ) || self.i128_type.is_compatible_with(typ))
    }

    pub fn is_native_int_type_or_bool(&self, typ: Type<'gcc>) -> bool {
        self.is_native_int_type(typ) || typ.is_compatible_with(self.bool_type)
    }

    pub fn is_int_type_or_bool(&self, typ: Type<'gcc>) -> bool {
        self.is_native_int_type(typ)
            || self.is_non_native_int_type(typ)
            || typ.is_compatible_with(self.bool_type)
    }

    pub fn sess(&self) -> &'tcx Session {
        self.tcx.sess
    }

    pub fn bitcast_if_needed(
        &self,
        value: RValue<'gcc>,
        expected_type: Type<'gcc>,
    ) -> RValue<'gcc> {
        if value.get_type() != expected_type {
            self.context.new_bitcast(None, value, expected_type)
        } else {
            value
        }
    }
}

impl<'gcc, 'tcx> BackendTypes for CodegenCx<'gcc, 'tcx> {
    type Value = RValue<'gcc>;
    type Metadata = RValue<'gcc>;
    // TODO(antoyo): change to Function<'gcc>.
    type Function = RValue<'gcc>;

    type BasicBlock = Block<'gcc>;
    type Type = Type<'gcc>;
    type Funclet = (); // TODO(antoyo)

    type DIScope = (); // TODO(antoyo)
    type DILocation = Location<'gcc>;
    type DIVariable = (); // TODO(antoyo)
}

impl<'gcc, 'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), RValue<'gcc>>> {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> RValue<'gcc> {
        let func = get_fn(self, instance);
        *self.current_func.borrow_mut() = Some(func);
        // FIXME(antoyo): this is a wrong cast. That requires changing the compiler API.
        unsafe { std::mem::transmute(func) }
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> RValue<'gcc> {
        let func_name = self.tcx.symbol_name(instance).name;

        let func = if self.intrinsics.borrow().contains_key(func_name) {
            self.intrinsics.borrow()[func_name]
        } else if let Some(variable) = self.get_declared_value(func_name) {
            return variable;
        } else {
            get_fn(self, instance)
        };
        let ptr = func.get_address(None);

        // TODO(antoyo): don't do this twice: i.e. in declare_fn and here.
        // FIXME(antoyo): the rustc API seems to call get_fn_addr() when not needed (e.g. for FFI).

        self.normal_function_addresses.borrow_mut().insert(ptr);
        self.function_address_names.borrow_mut().insert(ptr, func_name.to_string());

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
        let func = match tcx.lang_items().eh_personality() {
            Some(def_id) if !wants_msvc_seh(self.sess()) => {
                let instance = ty::Instance::expect_resolve(
                    tcx,
                    self.typing_env(),
                    def_id,
                    ty::List::empty(),
                    DUMMY_SP,
                );

                let symbol_name = tcx.symbol_name(instance).name;
                let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
                self.linkage.set(FunctionType::Extern);
                let func = self.declare_fn(symbol_name, fn_abi);
                let func: RValue<'gcc> = unsafe { std::mem::transmute(func) };
                func
            }
            _ => {
                let name = if wants_msvc_seh(self.sess()) {
                    "__CxxFrameHandler3"
                } else {
                    "rust_eh_personality"
                };
                let func = self.declare_func(name, self.type_i32(), &[], true);
                unsafe { std::mem::transmute::<Function<'gcc>, RValue<'gcc>>(func) }
            }
        };
        // TODO(antoyo): apply target cpu attributes.
        self.eh_personality.set(Some(func));
        func
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, _llfn: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn apply_target_cpu_attr(&self, _llfn: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        let entry_name = self.sess().target.entry_name.as_ref();
        if !self.functions.borrow().contains_key(entry_name) {
            #[cfg(feature = "master")]
            let conv = conv_to_fn_attribute(self.sess().target.entry_abi, &self.sess().target.arch);
            #[cfg(not(feature = "master"))]
            let conv = None;
            Some(self.declare_entry_fn(entry_name, fn_type, conv))
        } else {
            // If the symbol already exists, it is an error: for example, the user wrote
            // #[no_mangle] extern "C" fn main(..) {..}
            None
        }
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

impl<'gcc, 'tcx> HasWasmCAbiOpt for CodegenCx<'gcc, 'tcx> {
    fn wasm_c_abi_opt(&self) -> WasmCAbi {
        self.tcx.sess.opts.unstable_opts.wasm_c_abi
    }
}

impl<'gcc, 'tcx> HasX86AbiOpt for CodegenCx<'gcc, 'tcx> {
    fn x86_abi_opt(&self) -> X86Abi {
        X86Abi {
            regparm: self.tcx.sess.opts.unstable_opts.regparm,
            reg_struct_return: self.tcx.sess.opts.unstable_opts.reg_struct_return,
        }
    }
}

impl<'gcc, 'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'gcc, 'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_) | LayoutError::ReferencesError(_) = err {
            self.tcx.dcx().emit_fatal(respan(span, err.into_diagnostic()))
        } else {
            self.tcx.dcx().emit_fatal(ssa_errors::FailedToGetLayout { span, ty, err })
        }
    }
}

impl<'gcc, 'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'gcc, 'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_)) = err {
            self.tcx.dcx().emit_fatal(respan(span, err))
        } else {
            match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(span, "`fn_abi_of_fn_ptr({sig}, {extra_args:?})` failed: {err:?}");
                }
                FnAbiRequest::OfInstance { instance, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({instance}, {extra_args:?})` failed: {err:?}"
                    );
                }
            }
        }
    }
}

impl<'tcx, 'gcc> HasTypingEnv<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
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
        name.push('.');
        // Offset the index by the base so that always at least two characters
        // are generated. This avoids cases where the suffix is interpreted as
        // size by the assembler (for m68k: .b, .w, .l).
        name.push_str(&(idx as u64 + ALPHANUMERIC_ONLY as u64).to_base(ALPHANUMERIC_ONLY));
        name
    }
}

fn to_gcc_tls_mode(tls_model: TlsModel) -> gccjit::TlsModel {
    match tls_model {
        TlsModel::GeneralDynamic => gccjit::TlsModel::GlobalDynamic,
        TlsModel::LocalDynamic => gccjit::TlsModel::LocalDynamic,
        TlsModel::InitialExec => gccjit::TlsModel::InitialExec,
        TlsModel::LocalExec => gccjit::TlsModel::LocalExec,
        TlsModel::Emulated => gccjit::TlsModel::GlobalDynamic,
    }
}
