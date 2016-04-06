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

use rustc::hir::def_id::{DefId, DefIndex};
use syntax::codemap::Span;

use super::data::*;
use super::dump::Dump;
use super::span_utils::SpanUtils;

pub struct CsvDumper<'a, 'b, W: 'b> {
    output: &'b mut W,
    dump_spans: bool,
    span: SpanUtils<'a>
}

impl<'a, 'b, W: Write> CsvDumper<'a, 'b, W> {
    pub fn new(writer: &'b mut W, span: SpanUtils<'a>) -> CsvDumper<'a, 'b, W> {
        CsvDumper { output: writer, dump_spans: false, span: span }
    }

    fn record(&mut self, kind: &str, span: Span, values: String) {
        let span_str = self.span.extent_str(span);
        if let Err(_) = write!(self.output, "{},{}{}\n", kind, span_str, values) {
            error!("Error writing output");
        }
    }

    fn record_raw(&mut self, info: &str) {
        if let Err(_) = write!(self.output, "{}", info) {
            error!("Error writing output '{}'", info);
        }
    }

    pub fn dump_span(&mut self, kind: &str, span: Span) {
        assert!(self.dump_spans);
        let result = format!("span,kind,{},{},text,\"{}\"\n",
                             kind,
                             self.span.extent_str(span),
                             escape(self.span.snippet(span)));
        self.record_raw(&result);
    }
}

impl<'a, 'b, W: Write + 'b> Dump for CsvDumper<'a, 'b, W> {
    fn crate_prelude(&mut self, span: Span, data: CratePreludeData) {
        let crate_root = data.crate_root.unwrap_or("<no source>".to_owned());

        let values = make_values_str(&[
            ("name", &data.crate_name),
            ("crate_root", &crate_root)
        ]);

        self.record("crate", span, values);

        for c in data.external_crates {
            let num = c.num.to_string();
            let lo_loc = self.span.sess.codemap().lookup_char_pos(span.lo);
            let file_name = SpanUtils::make_path_string(&lo_loc.file.name);
            let values = make_values_str(&[
                ("name", &c.name),
                ("crate", &num),
                ("file_name", &file_name)
            ]);

            self.record_raw(&format!("external_crate{}\n", values));
        }

        self.record_raw("end_external_crates\n");
    }

    fn enum_data(&mut self, span: Span, data: EnumData) {
        if self.dump_spans {
            self.dump_span("enum", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("enum", data.span, values);
    }

    fn extern_crate(&mut self, span: Span, data: ExternCrateData) {
        if self.dump_spans {
            self.dump_span("extern_crate", span);
            return;
        }

        let id = data.id.to_string();
        let crate_num = data.crate_num.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("name", &data.name),
            ("location", &data.location),
            ("crate", &crate_num),
            ("scopeid", &scope)
        ]);

        self.record("extern_crate", data.span, values);
    }

    fn impl_data(&mut self, span: Span, data: ImplData) {
        if self.dump_spans {
            self.dump_span("impl", span);
            return;
        }

        let self_ref = data.self_ref.unwrap_or(null_def_id());
        let trait_ref = data.trait_ref.unwrap_or(null_def_id());

        let id = data.id.to_string();
        let ref_id = self_ref.index.as_usize().to_string();
        let ref_id_crate = self_ref.krate.to_string();
        let trait_id = trait_ref.index.as_usize().to_string();
        let trait_id_crate = trait_ref.krate.to_string();
        let scope = data.scope.to_string();
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
       if self.dump_spans {
           return;
       }

       let base_id = data.base_id.index.as_usize().to_string();
       let base_crate = data.base_id.krate.to_string();
       let deriv_id = data.deriv_id.to_string();
       let deriv_crate = 0.to_string();
       let values = make_values_str(&[
           ("base", &base_id),
           ("basecrate", &base_crate),
           ("derived", &deriv_id),
           ("derivedcrate", &deriv_crate)
       ]);

       self.record("inheritance", data.span, values);
    }

    fn function(&mut self, span: Span, data: FunctionData) {
        if self.dump_spans {
            self.dump_span("function", span);
            return;
        }

        let (decl_id, decl_crate) = match data.declaration {
            Some(id) => (id.index.as_usize().to_string(), id.krate.to_string()),
            None => (String::new(), String::new())
        };

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("declid", &decl_id),
            ("declidcrate", &decl_crate),
            ("scopeid", &scope)
        ]);

        self.record("function", data.span, values);
    }

    fn function_ref(&mut self, span: Span, data: FunctionRefData) {
        if self.dump_spans {
            self.dump_span("fn_ref", span);
            return;
        }

        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", ""),
            ("scopeid", &scope)
        ]);

        self.record("fn_ref", data.span, values);
    }

    fn function_call(&mut self, span: Span, data: FunctionCallData) {
        if self.dump_spans {
            self.dump_span("fn_call", span);
            return;
        }

        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let qualname = String::new();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &qualname),
            ("scopeid", &scope)
        ]);

        self.record("fn_call", data.span, values);
    }

    fn method(&mut self, span: Span, data: MethodData) {
        if self.dump_spans {
            self.dump_span("method_decl", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("method_decl", span, values);
    }

    fn method_call(&mut self, span: Span, data: MethodCallData) {
        if self.dump_spans {
            self.dump_span("method_call", span);
            return;
        }

        let (dcn, dck) = match data.decl_id {
            Some(declid) => (declid.index.as_usize().to_string(), declid.krate.to_string()),
            None => (String::new(), String::new()),
        };

        let ref_id = data.ref_id.unwrap_or(null_def_id());

        let def_id = ref_id.index.as_usize().to_string();
        let def_crate = ref_id.krate.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("refid", &def_id),
            ("refidcrate", &def_crate),
            ("declid", &dcn),
            ("declidcrate", &dck),
            ("scopeid", &scope)
        ]);

        self.record("method_call", data.span, values);
    }

    fn macro_data(&mut self, span: Span, data: MacroData) {
        if self.dump_spans {
            self.dump_span("macro", span);
            return;
        }

        let values = make_values_str(&[
            ("name", &data.name),
            ("qualname", &data.qualname)
        ]);

        self.record("macro", data.span, values);
    }

    fn macro_use(&mut self, span: Span, data: MacroUseData) {
        if self.dump_spans {
            self.dump_span("macro_use", span);
            return;
        }

        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("callee_name", &data.name),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("macro_use", data.span, values);
    }

    fn mod_data(&mut self, data: ModData) {
        if self.dump_spans {
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("def_file", &data.filename)
        ]);

        self.record("module", data.span, values);
    }

    fn mod_ref(&mut self, span: Span, data: ModRefData) {
        if self.dump_spans {
            self.dump_span("mod_ref", span);
            return;
        }

        let (ref_id, ref_crate) = match data.ref_id {
            Some(rid) => (rid.index.as_usize().to_string(), rid.krate.to_string()),
            None => (0.to_string(), 0.to_string())
        };

        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("mod_ref", data.span, values);
    }

    fn struct_data(&mut self, span: Span, data: StructData) {
        if self.dump_spans {
            self.dump_span("struct", span);
            return;
        }

        let id = data.id.to_string();
        let ctor_id = data.ctor_id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("ctor_id", &ctor_id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("struct", data.span, values);
    }

    fn struct_variant(&mut self, span: Span, data: StructVariantData) {
        if self.dump_spans {
            self.dump_span("variant_struct", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
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

    fn trait_data(&mut self, span: Span, data: TraitData) {
        if self.dump_spans {
            self.dump_span("trait", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("scopeid", &scope),
            ("value", &data.value)
        ]);

        self.record("trait", data.span, values);
    }

    fn tuple_variant(&mut self, span: Span, data: TupleVariantData) {
        if self.dump_spans {
            self.dump_span("variant", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
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

    fn type_ref(&mut self, span: Span, data: TypeRefData) {
        if self.dump_spans {
            self.dump_span("type_ref", span);
            return;
        }

        let (ref_id, ref_crate) = match data.ref_id {
            Some(id) => (id.index.as_usize().to_string(), id.krate.to_string()),
            None => (0.to_string(), 0.to_string())
        };

        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("qualname", &data.qualname),
            ("scopeid", &scope)
        ]);

        self.record("type_ref", data.span, values);
    }

    fn typedef(&mut self, span: Span, data: TypedefData) {
        if self.dump_spans {
            self.dump_span("typedef", span);
            return;
        }

        let id = data.id.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("qualname", &data.qualname),
            ("value", &data.value)
        ]);

        self.record("typedef", data.span, values);
    }

    fn use_data(&mut self, span: Span, data: UseData) {
        if self.dump_spans {
            self.dump_span("use_alias", span);
            return;
        }

        let mod_id = data.mod_id.unwrap_or(null_def_id());

        let id = data.id.to_string();
        let ref_id = mod_id.index.as_usize().to_string();
        let ref_crate = mod_id.krate.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("refid", &ref_id),
            ("refidcrate", &ref_crate),
            ("name", &data.name),
            ("scopeid", &scope)
        ]);

        self.record("use_alias", data.span, values);
    }

    fn use_glob(&mut self, span: Span, data: UseGlobData) {
        if self.dump_spans {
            self.dump_span("use_glob", span);
            return;
        }

        let names = data.names.join(", ");

        let id = data.id.to_string();
        let scope = data.scope.to_string();
        let values = make_values_str(&[
            ("id", &id),
            ("value", &names),
            ("scopeid", &scope)
        ]);

        self.record("use_glob", data.span, values);
    }

    fn variable(&mut self, span: Span, data: VariableData) {
        if self.dump_spans {
            self.dump_span("variable", span);
            return;
        }

        let id = data.id.to_string();
        let scope = data.scope.to_string();
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

    fn variable_ref(&mut self, span: Span, data: VariableRefData) {
        if self.dump_spans {
            self.dump_span("var_ref", span);
            return;
        }

        let ref_id = data.ref_id.index.as_usize().to_string();
        let ref_crate = data.ref_id.krate.to_string();
        let scope = data.scope.to_string();
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

fn null_def_id() -> DefId {
    DefId {
        krate: 0,
        index: DefIndex::new(0),
    }
}
