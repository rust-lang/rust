// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Row::*;

use super::escape;
use super::span_utils::SpanUtils;

use std::vec::Vec;

use syntax::ast;
use syntax::ast::{NodeId,DefId};
use syntax::codemap::*;

const ZERO_DEF_ID: DefId = DefId { node: 0, krate: 0 };

pub struct Recorder {
    // output file
    pub out: Box<Writer+'static>,
    pub dump_spans: bool,
}

impl Recorder {
    pub fn record(&mut self, info: &str) {
        match write!(self.out, "{}", info) {
            Err(_) => error!("Error writing output '{}'", info),
            _ => (),
        }
    }

    pub fn dump_span(&mut self,
                     su: SpanUtils,
                     kind: &str,
                     span: Span,
                     _sub_span: Option<Span>) {
        assert!(self.dump_spans);
        let result = format!("span,kind,{},{},text,\"{}\"\n",
                             kind, su.extent_str(span), escape(su.snippet(span)));
        self.record(&result[..]);
    }
}

pub struct FmtStrs<'a> {
    pub recorder: Box<Recorder>,
    span: SpanUtils<'a>,
}

macro_rules! s { ($e:expr) => { format!("{}", $e) }}
macro_rules! svec {
    ($($e:expr),*) => ({
        // leading _ to allow empty construction without a warning.
        let mut _temp = ::std::vec::Vec::new();
        $(_temp.push(s!($e));)*
        _temp
    })
}

#[derive(Copy, Debug, Eq, PartialEq)]
pub enum Row {
    Variable,
    Enum,
    Variant,
    VariantStruct,
    Function,
    MethodDecl,
    Struct,
    Trait,
    Impl,
    Module,
    UseAlias,
    UseGlob,
    ExternCrate,
    Inheritance,
    MethodCall,
    Typedef,
    ExternalCrate,
    Crate,
    FnCall,
    ModRef,
    VarRef,
    TypeRef,
    StructRef,
    FnRef,
}

impl<'a> FmtStrs<'a> {
    pub fn new(rec: Box<Recorder>, span: SpanUtils<'a>) -> FmtStrs<'a> {
        FmtStrs {
            recorder: rec,
            span: span,
        }
    }

    // A map from kind of item to a tuple of
    //   a string representation of the name
    //   a vector of field names
    //   whether this kind requires a span
    //   whether dump_spans should dump for this kind
    fn lookup_row(r: Row) -> (&'static str, Vec<&'static str>, bool, bool) {
        match r {
            Variable => ("variable",
                         vec!("id","name","qualname","value","type","scopeid"),
                         true, true),
            Enum => ("enum", vec!("id","qualname","scopeid","value"), true, true),
            Variant => ("variant",
                        vec!("id","name","qualname","type","value","scopeid"),
                        true, true),
            VariantStruct => ("variant_struct",
                              vec!("id","ctor_id","qualname","type","value","scopeid"),
                              true, true),
            Function => ("function",
                         vec!("id","qualname","declid","declidcrate","scopeid"),
                         true, true),
            MethodDecl => ("method_decl", vec!("id","qualname","scopeid"), true, true),
            Struct => ("struct", vec!("id","ctor_id","qualname","scopeid","value"), true, true),
            Trait => ("trait", vec!("id","qualname","scopeid","value"), true, true),
            Impl => ("impl",
                     vec!("id","refid","refidcrate","traitid","traitidcrate","scopeid"),
                     true, true),
            Module => ("module", vec!("id","qualname","scopeid","def_file"), true, false),
            UseAlias => ("use_alias",
                         vec!("id","refid","refidcrate","name","scopeid"),
                         true, true),
            UseGlob => ("use_glob", vec!("id","value","scopeid"), true, true),
            ExternCrate => ("extern_crate",
                            vec!("id","name","location","crate","scopeid"),
                            true, true),
            Inheritance => ("inheritance",
                            vec!("base","basecrate","derived","derivedcrate"),
                            true, false),
            MethodCall => ("method_call",
                           vec!("refid","refidcrate","declid","declidcrate","scopeid"),
                           true, true),
            Typedef => ("typedef", vec!("id","qualname","value"), true, true),
            ExternalCrate => ("external_crate", vec!("name","crate","file_name"), false, false),
            Crate => ("crate", vec!("name"), true, false),
            FnCall => ("fn_call", vec!("refid","refidcrate","qualname","scopeid"), true, true),
            ModRef => ("mod_ref", vec!("refid","refidcrate","qualname","scopeid"), true, true),
            VarRef => ("var_ref", vec!("refid","refidcrate","qualname","scopeid"), true, true),
            TypeRef => ("type_ref",
                        vec!("refid","refidcrate","qualname","scopeid"),
                        true, true),
            StructRef => ("struct_ref",
                          vec!("refid","refidcrate","qualname","scopeid"),
                          true, true),
            FnRef => ("fn_ref", vec!("refid","refidcrate","qualname","scopeid"), true, true)
        }
    }

    pub fn make_values_str(&self,
                           kind: &'static str,
                           fields: &Vec<&'static str>,
                           values: Vec<String>,
                           span: Span) -> Option<String> {
        if values.len() != fields.len() {
            self.span.sess.span_bug(span, &format!(
                "Mismatch between length of fields for '{}', expected '{}', found '{}'",
                kind, fields.len(), values.len())[]);
        }

        let values = values.iter().map(|s| {
            // Never take more than 1020 chars
            if s.len() > 1020 {
                &s[..1020]
            } else {
                &s[..]
            }
        });

        let pairs = fields.iter().zip(values);
        let strs = pairs.map(|(f, v)| format!(",{},\"{}\"", f, escape(String::from_str(v))));
        Some(strs.fold(String::new(), |mut s, ss| {
            s.push_str(&ss[..]);
            s
        }))
    }

    pub fn record_without_span(&mut self,
                               kind: Row,
                               values: Vec<String>,
                               span: Span) {
        let (label, ref fields, needs_span, dump_spans) = FmtStrs::lookup_row(kind);

        if needs_span {
            self.span.sess.span_bug(span, &format!(
                "Called record_without_span for '{}' which does requires a span",
                label)[]);
        }
        assert!(!dump_spans);

        if self.recorder.dump_spans {
            return;
        }

        let values_str = match self.make_values_str(label, fields, values, span) {
            Some(vs) => vs,
            None => return,
        };

        let mut result = String::from_str(label);
        result.push_str(&values_str[..]);
        result.push_str("\n");
        self.recorder.record(&result[..]);
    }

    pub fn record_with_span(&mut self,
                            kind: Row,
                            span: Span,
                            sub_span: Span,
                            values: Vec<String>) {
        let (label, ref fields, needs_span, dump_spans) = FmtStrs::lookup_row(kind);

        if self.recorder.dump_spans {
            if dump_spans {
                self.recorder.dump_span(self.span.clone(),
                                        label,
                                        span,
                                        Some(sub_span));
            }
            return;
        }

        if !needs_span {
            self.span.sess.span_bug(span,
                                    &format!("Called record_with_span for '{}' \
                                              which does not require a span", label));
        }

        let values_str = match self.make_values_str(label, fields, values, span) {
            Some(vs) => vs,
            None => return,
        };
        let result = format!("{},{}{}\n", label, self.span.extent_str(sub_span), values_str);
        self.recorder.record(&result[..]);
    }

    pub fn check_and_record(&mut self,
                            kind: Row,
                            span: Span,
                            sub_span: Option<Span>,
                            values: Vec<String>) {
        match sub_span {
            Some(sub_span) => self.record_with_span(kind, span, sub_span, values),
            None => {
                let (label, _, _, _) = FmtStrs::lookup_row(kind);
                self.span.report_span_err(label, span);
            }
        }
    }

    pub fn variable_str(&mut self,
                        span: Span,
                        sub_span: Option<Span>,
                        id: NodeId,
                        name: &str,
                        value: &str,
                        typ: &str) {
        // Getting a fully qualified name for a variable is hard because in
        // the local case they can be overridden in one block and there is no nice way
        // to refer to such a scope in english, so we just hack it by appending the
        // variable def's node id
        let mut qualname = String::from_str(name);
        qualname.push_str("$");
        qualname.push_str(&id.to_string()[]);
        self.check_and_record(Variable,
                              span,
                              sub_span,
                              svec!(id, name, qualname, value, typ, 0));
    }

    // formal parameters
    pub fn formal_str(&mut self,
                      span: Span,
                      sub_span: Option<Span>,
                      id: NodeId,
                      fn_name: &str,
                      name: &str,
                      typ: &str) {
        let mut qualname = String::from_str(fn_name);
        qualname.push_str("::");
        qualname.push_str(name);
        self.check_and_record(Variable,
                              span,
                              sub_span,
                              svec!(id, name, qualname, "", typ, 0));
    }

    // value is the initialising expression of the static if it is not mut, otherwise "".
    pub fn static_str(&mut self,
                      span: Span,
                      sub_span: Option<Span>,
                      id: NodeId,
                      name: &str,
                      qualname: &str,
                      value: &str,
                      typ: &str,
                      scope_id: NodeId) {
        self.check_and_record(Variable,
                              span,
                              sub_span,
                              svec!(id, name, qualname, value, typ, scope_id));
    }

    pub fn field_str(&mut self,
                     span: Span,
                     sub_span: Option<Span>,
                     id: NodeId,
                     name: &str,
                     qualname: &str,
                     typ: &str,
                     scope_id: NodeId) {
        self.check_and_record(Variable,
                              span,
                              sub_span,
                              svec!(id, name, qualname, "", typ, scope_id));
    }

    pub fn enum_str(&mut self,
                    span: Span,
                    sub_span: Option<Span>,
                    id: NodeId,
                    name: &str,
                    scope_id: NodeId,
                    value: &str) {
        self.check_and_record(Enum,
                              span,
                              sub_span,
                              svec!(id, name, scope_id, value));
    }

    pub fn tuple_variant_str(&mut self,
                             span: Span,
                             sub_span: Option<Span>,
                             id: NodeId,
                             name: &str,
                             qualname: &str,
                             typ: &str,
                             val: &str,
                             scope_id: NodeId) {
        self.check_and_record(Variant,
                              span,
                              sub_span,
                              svec!(id, name, qualname, typ, val, scope_id));
    }

    pub fn struct_variant_str(&mut self,
                              span: Span,
                              sub_span: Option<Span>,
                              id: NodeId,
                              ctor_id: NodeId,
                              name: &str,
                              typ: &str,
                              val: &str,
                              scope_id: NodeId) {
        self.check_and_record(VariantStruct,
                              span,
                              sub_span,
                              svec!(id, ctor_id, name, typ, val, scope_id));
    }

    pub fn fn_str(&mut self,
                  span: Span,
                  sub_span: Option<Span>,
                  id: NodeId,
                  name: &str,
                  scope_id: NodeId) {
        self.check_and_record(Function,
                              span,
                              sub_span,
                              svec!(id, name, "", "", scope_id));
    }

    pub fn method_str(&mut self,
                      span: Span,
                      sub_span: Option<Span>,
                      id: NodeId,
                      name: &str,
                      decl_id: Option<DefId>,
                      scope_id: NodeId) {
        let values = match decl_id {
            Some(decl_id) => svec!(id, name, decl_id.node, decl_id.krate, scope_id),
            None => svec!(id, name, "", "", scope_id)
        };
        self.check_and_record(Function,
                              span,
                              sub_span,
                              values);
    }

    pub fn method_decl_str(&mut self,
                           span: Span,
                           sub_span: Option<Span>,
                           id: NodeId,
                           name: &str,
                           scope_id: NodeId) {
        self.check_and_record(MethodDecl,
                              span,
                              sub_span,
                              svec!(id, name, scope_id));
    }

    pub fn struct_str(&mut self,
                      span: Span,
                      sub_span: Option<Span>,
                      id: NodeId,
                      ctor_id: NodeId,
                      name: &str,
                      scope_id: NodeId,
                      value: &str) {
        self.check_and_record(Struct,
                              span,
                              sub_span,
                              svec!(id, ctor_id, name, scope_id, value));
    }

    pub fn trait_str(&mut self,
                     span: Span,
                     sub_span: Option<Span>,
                     id: NodeId,
                     name: &str,
                     scope_id: NodeId,
                     value: &str) {
        self.check_and_record(Trait,
                              span,
                              sub_span,
                              svec!(id, name, scope_id, value));
    }

    pub fn impl_str(&mut self,
                    span: Span,
                    sub_span: Option<Span>,
                    id: NodeId,
                    ref_id: Option<DefId>,
                    trait_id: Option<DefId>,
                    scope_id: NodeId) {
        let ref_id = ref_id.unwrap_or(ZERO_DEF_ID);
        let trait_id = trait_id.unwrap_or(ZERO_DEF_ID);
        self.check_and_record(Impl,
                              span,
                              sub_span,
                              svec!(id,
                                    ref_id.node,
                                    ref_id.krate,
                                    trait_id.node,
                                    trait_id.krate,
                                    scope_id));
    }

    pub fn mod_str(&mut self,
                   span: Span,
                   sub_span: Option<Span>,
                   id: NodeId,
                   name: &str,
                   parent: NodeId,
                   filename: &str) {
        self.check_and_record(Module,
                              span,
                              sub_span,
                              svec!(id, name, parent, filename));
    }

    pub fn use_alias_str(&mut self,
                         span: Span,
                         sub_span: Option<Span>,
                         id: NodeId,
                         mod_id: Option<DefId>,
                         name: &str,
                         parent: NodeId) {
        let (mod_node, mod_crate) = match mod_id {
            Some(mod_id) => (mod_id.node, mod_id.krate),
            None => (0, 0)
        };
        self.check_and_record(UseAlias,
                              span,
                              sub_span,
                              svec!(id, mod_node, mod_crate, name, parent));
    }

    pub fn use_glob_str(&mut self,
                        span: Span,
                        sub_span: Option<Span>,
                        id: NodeId,
                        values: &str,
                        parent: NodeId) {
        self.check_and_record(UseGlob,
                              span,
                              sub_span,
                              svec!(id, values, parent));
    }

    pub fn extern_crate_str(&mut self,
                            span: Span,
                            sub_span: Option<Span>,
                            id: NodeId,
                            cnum: ast::CrateNum,
                            name: &str,
                            loc: &str,
                            parent: NodeId) {
        self.check_and_record(ExternCrate,
                              span,
                              sub_span,
                              svec!(id, name, loc, cnum, parent));
    }

    pub fn inherit_str(&mut self,
                       span: Span,
                       sub_span: Option<Span>,
                       base_id: DefId,
                       deriv_id: NodeId) {
        self.check_and_record(Inheritance,
                              span,
                              sub_span,
                              svec!(base_id.node,
                                    base_id.krate,
                                    deriv_id,
                                    0));
    }

    pub fn fn_call_str(&mut self,
                       span: Span,
                       sub_span: Option<Span>,
                       id: DefId,
                       scope_id:NodeId) {
        self.check_and_record(FnCall,
                              span,
                              sub_span,
                              svec!(id.node, id.krate, "", scope_id));
    }

    pub fn meth_call_str(&mut self,
                         span: Span,
                         sub_span: Option<Span>,
                         defid: Option<DefId>,
                         declid: Option<DefId>,
                         scope_id: NodeId) {
        let (dfn, dfk) = match defid {
            Some(defid) => (defid.node, defid.krate),
            None => (0, 0)
        };
        let (dcn, dck) = match declid {
            Some(declid) => (s!(declid.node), s!(declid.krate)),
            None => ("".to_string(), "".to_string())
        };
        self.check_and_record(MethodCall,
                              span,
                              sub_span,
                              svec!(dfn, dfk, dcn, dck, scope_id));
    }

    pub fn sub_mod_ref_str(&mut self,
                           span: Span,
                           sub_span: Span,
                           qualname: &str,
                           parent:NodeId) {
        self.record_with_span(ModRef,
                              span,
                              sub_span,
                              svec!(0, 0, qualname, parent));
    }

    pub fn typedef_str(&mut self,
                       span: Span,
                       sub_span: Option<Span>,
                       id: NodeId,
                       qualname: &str,
                       value: &str) {
        self.check_and_record(Typedef,
                              span,
                              sub_span,
                              svec!(id, qualname, value));
    }

    pub fn crate_str(&mut self,
                     span: Span,
                     name: &str) {
        self.record_with_span(Crate,
                              span,
                              span,
                              svec!(name));
    }

    pub fn external_crate_str(&mut self,
                              span: Span,
                              name: &str,
                              num: ast::CrateNum) {
        let lo_loc = self.span.sess.codemap().lookup_char_pos(span.lo);
        self.record_without_span(ExternalCrate,
                                 svec!(name, num, lo_loc.file.name),
                                 span);
    }

    pub fn sub_type_ref_str(&mut self,
                            span: Span,
                            sub_span: Span,
                            qualname: &str) {
        self.record_with_span(TypeRef,
                              span,
                              sub_span,
                              svec!(0, 0, qualname, 0));
    }

    // A slightly generic function for a reference to an item of any kind.
    pub fn ref_str(&mut self,
                   kind: Row,
                   span: Span,
                   sub_span: Option<Span>,
                   id: DefId,
                   scope_id: NodeId) {
        self.check_and_record(kind,
                              span,
                              sub_span,
                              svec!(id.node, id.krate, "", scope_id));
    }
}
