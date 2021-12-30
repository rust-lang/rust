use gccjit::{Function, FunctionType, GlobalKind, LValue, RValue, Type};
use rustc_codegen_ssa::traits::BaseTypeMethods;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::abi::call::FnAbi;

use crate::abi::FnAbiGccExt;
use crate::context::{CodegenCx, unit_name};
use crate::intrinsic::llvm;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn get_or_insert_global(&self, name: &str, ty: Type<'gcc>, is_tls: bool, link_section: Option<Symbol>) -> LValue<'gcc> {
        if self.globals.borrow().contains_key(name) {
            let typ = self.globals.borrow().get(name).expect("global").get_type();
            let global = self.context.new_global(None, GlobalKind::Imported, typ, name);
            if is_tls {
                global.set_tls_model(self.tls_model);
            }
            if let Some(link_section) = link_section {
                global.set_link_section(link_section.as_str());
            }
            global
        }
        else {
            self.declare_global(name, ty, is_tls, link_section)
        }
    }

    pub fn declare_unnamed_global(&self, ty: Type<'gcc>) -> LValue<'gcc> {
        let index = self.global_gen_sym_counter.get();
        self.global_gen_sym_counter.set(index + 1);
        let name = format!("global_{}_{}", index, unit_name(&self.codegen_unit));
        self.context.new_global(None, GlobalKind::Exported, ty, &name)
    }

    pub fn declare_global_with_linkage(&self, name: &str, ty: Type<'gcc>, linkage: GlobalKind) -> LValue<'gcc> {
        let global = self.context.new_global(None, linkage, ty, name);
        let global_address = global.get_address(None);
        self.globals.borrow_mut().insert(name.to_string(), global_address);
        global
    }

    /*pub fn declare_func(&self, name: &str, return_type: Type<'gcc>, params: &[Type<'gcc>], variadic: bool) -> RValue<'gcc> {
        self.linkage.set(FunctionType::Exported);
        let func = declare_raw_fn(self, name, () /*llvm::CCallConv*/, return_type, params, variadic);
        // FIXME(antoyo): this is a wrong cast. That requires changing the compiler API.
        unsafe { std::mem::transmute(func) }
    }*/

    pub fn declare_global(&self, name: &str, ty: Type<'gcc>, is_tls: bool, link_section: Option<Symbol>) -> LValue<'gcc> {
        let global = self.context.new_global(None, GlobalKind::Exported, ty, name);
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

    pub fn declare_cfn(&self, name: &str, _fn_type: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use the fn_type parameter.
        let const_string = self.context.new_type::<u8>().make_pointer().make_pointer();
        let return_type = self.type_i32();
        let variadic = false;
        self.linkage.set(FunctionType::Exported);
        let func = declare_raw_fn(self, name, () /*llvm::CCallConv*/, return_type, &[self.type_i32(), const_string], variadic);
        // NOTE: it is needed to set the current_func here as well, because get_fn() is not called
        // for the main function.
        *self.current_func.borrow_mut() = Some(func);
        // FIXME(antoyo): this is a wrong cast. That requires changing the compiler API.
        unsafe { std::mem::transmute(func) }
    }

    pub fn declare_fn(&self, name: &str, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> RValue<'gcc> {
        let (return_type, params, variadic) = fn_abi.gcc_type(self);
        let func = declare_raw_fn(self, name, () /*fn_abi.llvm_cconv()*/, return_type, &params, variadic);
        // FIXME(antoyo): this is a wrong cast. That requires changing the compiler API.
        unsafe { std::mem::transmute(func) }
    }

    pub fn define_global(&self, name: &str, ty: Type<'gcc>, is_tls: bool, link_section: Option<Symbol>) -> LValue<'gcc> {
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
fn declare_raw_fn<'gcc>(cx: &CodegenCx<'gcc, '_>, name: &str, _callconv: () /*llvm::CallConv*/, return_type: Type<'gcc>, param_types: &[Type<'gcc>], variadic: bool) -> Function<'gcc> {
    if name.starts_with("llvm.") {
        return llvm::intrinsic(name, cx);
    }
    let func =
        if cx.functions.borrow().contains_key(name) {
            *cx.functions.borrow().get(name).expect("function")
        }
        else {
            let params: Vec<_> = param_types.into_iter().enumerate()
                .map(|(index, param)| cx.context.new_parameter(None, *param, &format!("param{}", index))) // TODO(antoyo): set name.
                .collect();
            let func = cx.context.new_function(None, cx.linkage.get(), return_type, &params, mangle_name(name), variadic);
            cx.functions.borrow_mut().insert(name.to_string(), func);
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
// Unsupported characters: `$` and `.`.
pub fn mangle_name(name: &str) -> String {
    name.replace(|char: char| {
        if !char.is_alphanumeric() && char != '_' {
            debug_assert!("$.".contains(char), "Unsupported char in function name: {}", char);
            true
        }
        else {
            false
        }
    }, "_")
}
