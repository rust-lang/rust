// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::external_data::*;

pub trait Dump {
    fn crate_prelude(&mut self, CratePreludeData) {}
    fn enum_data(&mut self, EnumData) {}
    fn extern_crate(&mut self, ExternCrateData) {}
    fn impl_data(&mut self, ImplData) {}
    fn inheritance(&mut self, InheritanceData) {}
    fn function(&mut self, FunctionData) {}
    fn function_ref(&mut self, FunctionRefData) {}
    fn function_call(&mut self, FunctionCallData) {}
    fn method(&mut self, MethodData) {}
    fn method_call(&mut self, MethodCallData) {}
    fn macro_data(&mut self, MacroData) {}
    fn macro_use(&mut self, MacroUseData) {}
    fn mod_data(&mut self, ModData) {}
    fn mod_ref(&mut self, ModRefData) {}
    fn struct_data(&mut self, StructData) {}
    fn struct_variant(&mut self, StructVariantData) {}
    fn trait_data(&mut self, TraitData) {}
    fn tuple_variant(&mut self, TupleVariantData) {}
    fn type_ref(&mut self, TypeRefData) {}
    fn typedef(&mut self, TypeDefData) {}
    fn use_data(&mut self, UseData) {}
    fn use_glob(&mut self, UseGlobData) {}
    fn variable(&mut self, VariableData) {}
    fn variable_ref(&mut self, VariableRefData) {}
}
