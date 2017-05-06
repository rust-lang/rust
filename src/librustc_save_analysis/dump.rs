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

use rls_data::CratePreludeData;

pub trait Dump {
    fn crate_prelude(&mut self, _: CratePreludeData) {}
    fn enum_data(&mut self, _: EnumData) {}
    fn extern_crate(&mut self, _: ExternCrateData) {}
    fn impl_data(&mut self, _: ImplData) {}
    fn inheritance(&mut self, _: InheritanceData) {}
    fn function(&mut self, _: FunctionData) {}
    fn function_ref(&mut self, _: FunctionRefData) {}
    fn function_call(&mut self, _: FunctionCallData) {}
    fn method(&mut self, _: MethodData) {}
    fn method_call(&mut self, _: MethodCallData) {}
    fn macro_data(&mut self, _: MacroData) {}
    fn macro_use(&mut self, _: MacroUseData) {}
    fn mod_data(&mut self, _: ModData) {}
    fn mod_ref(&mut self, _: ModRefData) {}
    fn struct_data(&mut self, _: StructData) {}
    fn struct_variant(&mut self, _: StructVariantData) {}
    fn trait_data(&mut self, _: TraitData) {}
    fn tuple_variant(&mut self, _: TupleVariantData) {}
    fn type_ref(&mut self, _: TypeRefData) {}
    fn typedef(&mut self, _: TypeDefData) {}
    fn use_data(&mut self, _: UseData) {}
    fn use_glob(&mut self, _: UseGlobData) {}
    fn variable(&mut self, _: VariableData) {}
    fn variable_ref(&mut self, _: VariableRefData) {}
}
