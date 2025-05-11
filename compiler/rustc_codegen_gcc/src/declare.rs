#[cfg(feature = "master")]
use gccjit::{FnAttribute, ToRValue};
use gccjit::{Function, FunctionType, GlobalKind, LValue, RValue, Type};
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::abi::{FnAbiGcc, FnAbiGccExt};
use crate::context::CodegenCx;
use crate::intrinsic::llvm;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn get_or_insert_global(
        &self,
        name: &str,
        ty: Type<'gcc>,
        is_tls: bool,
        link_section: Option<Symbol>,
    ) -> LValue<'gcc> {
        if self.globals.borrow().contains_key(name) {
            let typ = self.globals.borrow()[name].get_type();
            let global = self.context.new_global(None, GlobalKind::Imported, typ, name);
            if is_tls {
                global.set_tls_model(self.tls_model);
            }
            if let Some(link_section) = link_section {
                global.set_link_section(link_section.as_str());
            }
            global
        } else {
            self.declare_global(name, ty, GlobalKind::Exported, is_tls, link_section)
        }
    }

    pub fn declare_unnamed_global(&self, ty: Type<'gcc>) -> LValue<'gcc> {
        let name = self.generate_local_symbol_name("global");
        self.context.new_global(None, GlobalKind::Internal, ty, name)
    }

    pub fn declare_global_with_linkage(
        &self,
        name: &str,
        ty: Type<'gcc>,
        linkage: GlobalKind,
    ) -> LValue<'gcc> {
        let global = self.context.new_global(None, linkage, ty, name);
        let global_address = global.get_address(None);
        self.globals.borrow_mut().insert(name.to_string(), global_address);
        global
    }

    pub fn declare_func(
        &self,
        name: &str,
        return_type: Type<'gcc>,
        params: &[Type<'gcc>],
        variadic: bool,
    ) -> Function<'gcc> {
        self.linkage.set(FunctionType::Extern);
        declare_raw_fn(self, name, None, return_type, params, variadic)
    }

    pub fn declare_global(
        &self,
        name: &str,
        ty: Type<'gcc>,
        global_kind: GlobalKind,
        is_tls: bool,
        link_section: Option<Symbol>,
    ) -> LValue<'gcc> {
        let global = self.context.new_global(None, global_kind, ty, name);
        if is_tls {
            global.set_tls_model(self.tls_model);
        }
        if let Some(link_section) = link_section {
            global.set_link_section(link_section.as_str());
        }
        let global_address = global.get_address(None);
        self.globals.borrow_mut().insert(name.to_string(), global_address);
        global
    }

    pub fn declare_private_global(&self, name: &str, ty: Type<'gcc>) -> LValue<'gcc> {
        let global = self.context.new_global(None, GlobalKind::Internal, ty, name);
        let global_address = global.get_address(None);
        self.globals.borrow_mut().insert(name.to_string(), global_address);
        global
    }

    pub fn declare_entry_fn(
        &self,
        name: &str,
        _fn_type: Type<'gcc>,
        #[cfg(feature = "master")] callconv: Option<FnAttribute<'gcc>>,
        #[cfg(not(feature = "master"))] callconv: Option<()>,
    ) -> RValue<'gcc> {
        // TODO(antoyo): use the fn_type parameter.
        let const_string = self.context.new_type::<u8>().make_pointer().make_pointer();
        let return_type = self.type_i32();
        let variadic = false;
        self.linkage.set(FunctionType::Exported);
        let func = declare_raw_fn(
            self,
            name,
            callconv,
            return_type,
            &[self.type_i32(), const_string],
            variadic,
        );
        // NOTE: it is needed to set the current_func here as well, because get_fn() is not called
        // for the main function.
        *self.current_func.borrow_mut() = Some(func);
        // FIXME(antoyo): this is a wrong cast. That requires changing the compiler API.
        unsafe { std::mem::transmute(func) }
    }

    pub fn declare_fn(&self, name: &str, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Function<'gcc> {
        let FnAbiGcc {
            return_type,
            arguments_type,
            is_c_variadic,
            on_stack_param_indices,
            #[cfg(feature = "master")]
            fn_attributes,
        } = fn_abi.gcc_type(self);
        #[cfg(feature = "master")]
        let conv = fn_abi.gcc_cconv(self);
        #[cfg(not(feature = "master"))]
        let conv = None;
        let func = declare_raw_fn(self, name, conv, return_type, &arguments_type, is_c_variadic);
        self.on_stack_function_params.borrow_mut().insert(func, on_stack_param_indices);
        #[cfg(feature = "master")]
        for fn_attr in fn_attributes {
            func.add_attribute(fn_attr);
        }
        func
    }

    pub fn define_global(
        &self,
        name: &str,
        ty: Type<'gcc>,
        is_tls: bool,
        link_section: Option<Symbol>,
    ) -> LValue<'gcc> {
        self.get_or_insert_global(name, ty, is_tls, link_section)
    }

    pub fn get_declared_value(&self, name: &str) -> Option<RValue<'gcc>> {
        // TODO(antoyo): use a different field than globals, because this seems to return a function?
        self.globals.borrow().get(name).cloned()
    }
}

/// Declare a function.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing Value instead.
fn declare_raw_fn<'gcc>(
    cx: &CodegenCx<'gcc, '_>,
    name: &str,
    #[cfg(feature = "master")] callconv: Option<FnAttribute<'gcc>>,
    #[cfg(not(feature = "master"))] _callconv: Option<()>,
    return_type: Type<'gcc>,
    param_types: &[Type<'gcc>],
    variadic: bool,
) -> Function<'gcc> {
    if name.starts_with("llvm.") {
        let intrinsic = match name {
            "llvm.fma.f16" => {
                // fma is not a target builtin, but a normal builtin, so we handle it differently
                // here.
                cx.context.get_builtin_function("fma")
            }
            _ => llvm::intrinsic(name, cx),
        };

        cx.intrinsics.borrow_mut().insert(name.to_string(), intrinsic);
        return intrinsic;
    }
    let func = if cx.functions.borrow().contains_key(name) {
        cx.functions.borrow()[name]
    } else {
        let params: Vec<_> = param_types
            .iter()
            .enumerate()
            .map(|(index, param)| cx.context.new_parameter(None, *param, format!("param{}", index))) // TODO(antoyo): set name.
            .collect();
        #[cfg(not(feature = "master"))]
        let name = &mangle_name(name);
        let func =
            cx.context.new_function(None, cx.linkage.get(), return_type, &params, name, variadic);
        #[cfg(feature = "master")]
        if let Some(attribute) = callconv {
            func.add_attribute(attribute);
        }
        cx.functions.borrow_mut().insert(name.to_string(), func);

        #[cfg(feature = "master")]
        if name == "rust_eh_personality" {
            // NOTE: GCC will sometimes change the personality function set on a function from
            // rust_eh_personality to __gcc_personality_v0 as an optimization.
            // As such, we need to create a weak alias from __gcc_personality_v0 to
            // rust_eh_personality in order to avoid a linker error.
            // This needs to be weak in order to still allow using the standard
            // __gcc_personality_v0 when the linking to it.
            // Since aliases don't work (maybe because of a bug in LTO partitioning?), we
            // create a wrapper function that calls rust_eh_personality.

            let params: Vec<_> = param_types
                .iter()
                .enumerate()
                .map(|(index, param)| {
                    cx.context.new_parameter(None, *param, format!("param{}", index))
                }) // TODO(antoyo): set name.
                .collect();
            let gcc_func = cx.context.new_function(
                None,
                FunctionType::Exported,
                return_type,
                &params,
                "__gcc_personality_v0",
                variadic,
            );

            // We need a normal extern function for the crates that access rust_eh_personality
            // without defining it, otherwise we'll get a compiler error.
            //
            // For the crate defining it, that needs to be a weak alias instead.
            gcc_func.add_attribute(FnAttribute::Weak);

            let block = gcc_func.new_block("start");
            let mut args = vec![];
            for param in &params {
                args.push(param.to_rvalue());
            }
            let call = cx.context.new_call(None, func, &args);
            if return_type == cx.type_void() {
                block.add_eval(None, call);
                block.end_with_void_return(None);
            } else {
                block.end_with_return(None, call);
            }
        }

        func
    };

    // TODO(antoyo): set function calling convention.
    // TODO(antoyo): set unnamed address.
    // TODO(antoyo): set no red zone function attribute.
    // TODO(antoyo): set attributes for optimisation.
    // TODO(antoyo): set attributes for non lazy bind.

    // FIXME(antoyo): invalid cast.
    func
}

// FIXME(antoyo): this is a hack because libgccjit currently only supports alpha, num and _.
// Unsupported characters: `$`, `.` and `*`.
// FIXME(antoyo): `*` might not be expected: https://github.com/rust-lang/rust/issues/116979#issuecomment-1840926865
#[cfg(not(feature = "master"))]
fn mangle_name(name: &str) -> String {
    name.replace(
        |char: char| {
            if !char.is_alphanumeric() && char != '_' {
                debug_assert!(
                    "$.*".contains(char),
                    "Unsupported char in function name {}: {}",
                    name,
                    char
                );
                true
            } else {
                false
            }
        },
        "_",
    )
}
