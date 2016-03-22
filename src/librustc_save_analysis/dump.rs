// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::codemap::Span;

use super::data::*;

pub trait Dump {
    fn crate_prelude(&mut self, _: Span, _: CratePreludeData) {}
    fn enum_data(&mut self, _: Span, _: EnumData) {}
    fn extern_crate(&mut self, _: Span, _: ExternCrateData) {}
    fn impl_data(&mut self, _: Span, _: ImplData) {}
    fn inheritance(&mut self, _: InheritanceData) {}
    fn function(&mut self, _: Span, _: FunctionData) {}
    fn function_ref(&mut self, _: Span, _: FunctionRefData) {}
    fn function_call(&mut self, _: Span, _: FunctionCallData) {}
    fn method(&mut self, _: Span, _: MethodData) {}
    fn method_call(&mut self, _: Span, _: MethodCallData) {}
    fn macro_data(&mut self, _: Span, _: MacroData) {}
    fn macro_use(&mut self, _: Span, _: MacroUseData) {}
    fn mod_data(&mut self, _: ModData) {}
    fn mod_ref(&mut self, _: Span, _: ModRefData) {}
    fn struct_data(&mut self, _: Span, _: StructData) {}
    fn struct_variant(&mut self, _: Span, _: StructVariantData) {}
    fn trait_data(&mut self, _: Span, _: TraitData) {}
    fn tuple_variant(&mut self, _: Span, _: TupleVariantData) {}
    fn type_ref(&mut self, _: Span, _: TypeRefData) {}
    fn typedef(&mut self, _: Span, _: TypedefData) {}
    fn use_data(&mut self, _: Span, _: UseData) {}
    fn use_glob(&mut self, _: Span, _: UseGlobData) {}
    fn variable(&mut self, _: Span, _: VariableData) {}
    fn variable_ref(&mut self, _: Span, _: VariableRefData) {}
}
