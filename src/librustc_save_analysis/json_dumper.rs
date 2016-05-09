// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Write;

use rustc_serialize::json::as_json;

use super::external_data::*;
use super::dump::Dump;

pub struct JsonDumper<'b, W: Write + 'b> {
    output: &'b mut W,
    first: bool,
}

impl<'b, W: Write> JsonDumper<'b, W> {
    pub fn new(writer: &'b mut W) -> JsonDumper<'b, W> {
        if let Err(_) = write!(writer, "[") {
            error!("Error writing output");
        }
        JsonDumper { output: writer, first: true }
    }
}

impl<'b, W: Write> Drop for JsonDumper<'b, W> {
    fn drop(&mut self) {
        if let Err(_) = write!(self.output, "]") {
            error!("Error writing output");
        }
    }
}

macro_rules! impl_fn {
    ($fn_name: ident, $data_type: ident) => {
        fn $fn_name(&mut self, data: $data_type) {
            if self.first {
                self.first = false;
            } else {
                if let Err(_) = write!(self.output, ",") {
                    error!("Error writing output");
                }
            }
            if let Err(_) = write!(self.output, "{}", as_json(&data)) {
                error!("Error writing output '{}'", as_json(&data));
            }
        }
    }
}

impl<'b, W: Write + 'b> Dump for JsonDumper<'b, W> {
    impl_fn!(crate_prelude, CratePreludeData);
    impl_fn!(enum_data, EnumData);
    impl_fn!(extern_crate, ExternCrateData);
    impl_fn!(impl_data, ImplData);
    impl_fn!(inheritance, InheritanceData);
    impl_fn!(function, FunctionData);
    impl_fn!(function_ref, FunctionRefData);
    impl_fn!(function_call, FunctionCallData);
    impl_fn!(method, MethodData);
    impl_fn!(method_call, MethodCallData);
    impl_fn!(macro_data, MacroData);
    impl_fn!(macro_use, MacroUseData);
    impl_fn!(mod_data, ModData);
    impl_fn!(mod_ref, ModRefData);
    impl_fn!(struct_data, StructData);
    impl_fn!(struct_variant, StructVariantData);
    impl_fn!(trait_data, TraitData);
    impl_fn!(tuple_variant, TupleVariantData);
    impl_fn!(type_ref, TypeRefData);
    impl_fn!(typedef, TypedefData);
    impl_fn!(use_data, UseData);
    impl_fn!(use_glob, UseGlobData);
    impl_fn!(variable, VariableData);
    impl_fn!(variable_ref, VariableRefData);
}
