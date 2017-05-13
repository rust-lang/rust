// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Write;

use super::external_data::*;
use super::dump::Dump;

pub struct CsvDumper<'b, W: 'b> {
    output: &'b mut W
}

impl<'b, W: Write> CsvDumper<'b, W> {
    pub fn new(writer: &'b mut W) -> CsvDumper<'b, W> {
        CsvDumper { output: writer }
    }

    fn record(&mut self, kind: &str, span: SpanData, values: String) {
        let span_str = span_extent_str(span);
        if let Err(_) = write!(self.output, "{},{}{}\n", kind, span_str, values) {
            error!("Error writing output");
        }
    }

    fn record_raw(&mut self, info: &str) {
        if let Err(_) = write!(self.output, "{}", info) {
            error!("Error writing output '{}'", info);
        }
    }
}

impl<'b, W: Write + 'b> Dump for CsvDumper<'b, W> {
    fn crate_prelude(&mut self, data: CratePreludeData) {
        let values = make_values_str(&[
            ("name", &data.crate_name),
            ("crate_root", &data.crate_root)
        ]);

        self.record("crate", data.span, values);

        for c in data.external_crates {
            let num = c.num.to_string();
            let values = make_values_str(&[
                ("name", &c.name),
                ("crate", &num),
                ("file_name", &c.file_name)
            ]);

            self.record_raw(&format!("external_crate{}\n", values));
        }

        self.record_raw("end_external_crates\n");
    }

    fn enum_data(&mut self, data: EnumData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("enum", data.span, values);
    }

    fn extern_crate(&mut self, data: ExternCrateData) {
        let id = data.id.index.as_u32().to_string();
        let crate_num = data.crate_num.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("name", &data.name),
            ("location", &data.location),
            ("crate", &crate_num),
            ("scopeid", &scope)
        ]);

        self.record("extern_crate", data.span, values);
    }

    fn impl_data(&mut self, data: ImplData) {
        let self_ref = data.self_ref.unwrap_or(null_def_id());
        let trait_ref = data.trait_ref.unwrap_or(null_def_id());

        let id = data.id.index.as_u32().to_string();
        let ref_id = self_ref.index.as_usize().to_string();
        let ref_id_crate = self_ref.krate.to_string();
        let trait_id = trait_ref.index.as_usize().to_string();
        let trait_id_crate = trait_ref.krate.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("refid", &ref_id),
            ("refidcrate", &ref_id_crate),
            ("traitid", &trait_id),
            ("traitidcrate", &trait_id_crate),
            ("scopeid", &scope)
        ]);

        self.record("impl", data.span, values);
    }

    fn inheritance(&mut self, data: InheritanceData) {
       let base_id = data.base_id.index.as_usize().to_string();
       let base_crate = data.base_id.krate.to_string();
       let deriv_id = data.deriv_id.index.as_u32().to_string();
       let deriv_crate = data.deriv_id.krate.to_string();
       let values = make_values_str(&[
           ("base", &base_id),
           ("basecrate", &base_crate),
           ("derived", &deriv_id),
           ("derivedcrate", &deriv_crate)
       ]);

       self.record("inheritance", data.span, values);
    }

    fn function(&mut self, data: FunctionData) {
        let (decl_id, decl_crate) = match data.declaration {
            Some(id) => (id.index.as_usize().to_string(), id.krate.to_string()),
            None => (String::new(), String::new())
        };

        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("declid", &decl_id),
            ("declidcrate", &decl_crate),
            ("scopeid", &scope)
        ]);

        self.record("function", data.span, values);
    }

    fn function_ref(&mut self, data: FunctionRefData) {
        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", ""),
            ("scopeid", &scope)
        ]);

        self.record("fn_ref", data.span, values);
    }

    fn function_call(&mut self, data: FunctionCallData) {
        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let qualname = String::new();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &qualname),
            ("scopeid", &scope)
        ]);

        self.record("fn_call", data.span, values);
    }

    fn method(&mut self, data: MethodData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("method_decl", data.span, values);
    }

    fn method_call(&mut self, data: MethodCallData) {
        let (dcn, dck) = match data.decl_id {
            Some(declid) => (declid.index.as_usize().to_string(), declid.krate.to_string()),
            None => (String::new(), String::new()),
        };

        let ref_id = data.ref_id.unwrap_or(null_def_id());

        let def_id = ref_id.index.as_usize().to_string();
        let def_crate = ref_id.krate.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &def_id),
            ("refidcrate", &def_crate),
            ("declid", &dcn),
            ("declidcrate", &dck),
            ("scopeid", &scope)
        ]);

        self.record("method_call", data.span, values);
    }

    fn macro_data(&mut self, data: MacroData) {
        let values = make_values_str(&[
            ("name", &data.name),
            ("qualname", &data.qualname)
        ]);

        self.record("macro", data.span, values);
    }

    fn macro_use(&mut self, data: MacroUseData) {
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("callee_name", &data.name),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("macro_use", data.span, values);
    }

    fn mod_data(&mut self, data: ModData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("def_file", &data.filename)
        ]);

        self.record("module", data.span, values);
    }

    fn mod_ref(&mut self, data: ModRefData) {
        let (ref_id, ref_crate) = match data.ref_id {
            Some(rid) => (rid.index.as_usize().to_string(), rid.krate.to_string()),
            None => (0.to_string(), 0.to_string())
        };

        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("mod_ref", data.span, values);
    }

    fn struct_data(&mut self, data: StructData) {
        let id = data.id.index.as_u32().to_string();
        let ctor_id = data.ctor_id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("ctor_id", &ctor_id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("struct", data.span, values);
    }

    fn struct_variant(&mut self, data: StructVariantData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("ctor_id", &id),
            ("qualname", &data.qualname),
            ("type", &data.type_value),
            ("value", &data.value),
            ("scopeid", &scope)
        ]);

        self.record("variant_struct", data.span, values);
    }

    fn trait_data(&mut self, data: TraitData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("trait", data.span, values);
    }

    fn tuple_variant(&mut self, data: TupleVariantData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("name", &data.name),
            ("qualname", &data.qualname),
            ("type", &data.type_value),
            ("value", &data.value),
            ("scopeid", &scope)
        ]);

        self.record("variant", data.span, values);
    }

    fn type_ref(&mut self, data: TypeRefData) {
        let (ref_id, ref_crate) = match data.ref_id {
            Some(id) => (id.index.as_usize().to_string(), id.krate.to_string()),
            None => (0.to_string(), 0.to_string())
        };

        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("type_ref", data.span, values);
    }

    fn typedef(&mut self, data: TypeDefData) {
        let id = data.id.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("value", &data.value)
        ]);

        self.record("typedef", data.span, values);
    }

    fn use_data(&mut self, data: UseData) {
        let mod_id = data.mod_id.unwrap_or(null_def_id());

        let id = data.id.index.as_u32().to_string();
        let ref_id = mod_id.index.as_usize().to_string();
        let ref_crate = mod_id.krate.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("name", &data.name),
            ("scopeid", &scope)
        ]);

        self.record("use_alias", data.span, values);
    }

    fn use_glob(&mut self, data: UseGlobData) {
        let names = data.names.join(", ");

        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("value", &names),
            ("scopeid", &scope)
        ]);

        self.record("use_glob", data.span, values);
    }

    fn variable(&mut self, data: VariableData) {
        let id = data.id.index.as_u32().to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("name", &data.name),
            ("qualname", &data.qualname),
            ("value", &data.value),
            ("type", &data.type_value),
            ("scopeid", &scope)
        ]);

        self.record("variable", data.span, values);
    }

    fn variable_ref(&mut self, data: VariableRefData) {
        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let scope = data.scope.index.as_u32().to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", ""),
            ("scopeid", &scope)
        ]);

        self.record("var_ref", data.span, values)
    }
}

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

fn make_values_str(pairs: &[(&'static str, &str)]) -> String {
    let pairs = pairs.into_iter().map(|&(f, v)| {
        // Never take more than 1020 chars
        if v.len() > 1020 {
            (f, &v[..1020])
        } else {
            (f, v)
        }
    });

    let strs = pairs.map(|(f, v)| format!(",{},\"{}\"", f, escape(String::from(v))));
    strs.fold(String::new(), |mut s, ss| {
        s.push_str(&ss[..]);
        s
    })
}

fn span_extent_str(span: SpanData) -> String {
    format!("file_name,\"{}\",file_line,{},file_col,{},byte_start,{},\
             file_line_end,{},file_col_end,{},byte_end,{}",
             span.file_name, span.line_start, span.column_start, span.byte_start,
             span.line_end, span.column_end, span.byte_end)
}
