//! Write the output of rustc's analysis to an implementor of Dump.
//!
//! Dumping the analysis is implemented by walking the AST and getting a bunch of
//! info out from all over the place. We use `DefId`s to identify objects. The
//! tricky part is getting syntactic (span, source text) and semantic (reference
//! `DefId`s) information for parts of expressions which the compiler has discarded.
//! E.g., in a path `foo::bar::baz`, the compiler only keeps a span for the whole
//! path and a reference to `baz`, but we want spans and references for all three
//! idents.
//!
//! SpanUtils is used to manipulate spans. In particular, to extract sub-spans
//! from spans (e.g., the span for `bar` from the above example path).
//! DumpVisitor walks the AST and processes it, and JsonDumper is used for
//! recording the output.

use rustc::hir::def::{Res, DefKind as HirDefKind};
use rustc::hir::def_id::DefId;
use rustc::session::config::Input;
use rustc::span_bug;
use rustc::ty::{self, DefIdTree, TyCtxt};
use rustc_data_structures::fx::FxHashSet;

use std::path::Path;
use std::env;

use syntax::ast::{self, Attribute, NodeId, PatKind, CRATE_NODE_ID};
use syntax::parse::token;
use syntax::visit::{self, Visitor};
use syntax::print::pprust::{
    bounds_to_string,
    generic_params_to_string,
    ty_to_string
};
use syntax::ptr::P;
use syntax::source_map::{Spanned, DUMMY_SP, respan};
use syntax::walk_list;
use syntax_pos::*;

use crate::{escape, generated_code, id_from_def_id, id_from_node_id, lower_attributes,
            PathCollector, SaveContext};
use crate::json_dumper::{Access, DumpOutput, JsonDumper};
use crate::span_utils::SpanUtils;
use crate::sig;

use rls_data::{CompilationOptions, CratePreludeData, Def, DefKind, GlobalCrateId, Import,
               ImportKind, Ref, RefKind, Relation, RelationKind, SpanData};

use log::{debug, error};

macro_rules! down_cast_data {
    ($id:ident, $kind:ident, $sp:expr) => {
        let $id = if let super::Data::$kind(data) = $id {
            data
        } else {
            span_bug!($sp, "unexpected data kind: {:?}", $id);
        };
    };
}

macro_rules! access_from {
    ($save_ctxt:expr, $item:expr, $id:expr) => {
        Access {
            public: $item.vis.node.is_pub(),
            reachable: $save_ctxt.access_levels.is_reachable($id),
        }
    };
}

macro_rules! access_from_vis {
    ($save_ctxt:expr, $vis:expr, $id:expr) => {
        Access {
            public: $vis.node.is_pub(),
            reachable: $save_ctxt.access_levels.is_reachable($id),
        }
    };
}

pub struct DumpVisitor<'l, 'tcx, 'll, O: DumpOutput> {
    save_ctxt: SaveContext<'l, 'tcx>,
    tcx: TyCtxt<'tcx>,
    dumper: &'ll mut JsonDumper<O>,

    span: SpanUtils<'l>,

    cur_scope: NodeId,

    // Set of macro definition (callee) spans, and the set
    // of macro use (callsite) spans. We store these to ensure
    // we only write one macro def per unique macro definition, and
    // one macro use per unique callsite span.
    // mac_defs: FxHashSet<Span>,
    // macro_calls: FxHashSet<Span>,
}

impl<'l, 'tcx, 'll, O: DumpOutput + 'll> DumpVisitor<'l, 'tcx, 'll, O> {
    pub fn new(
        save_ctxt: SaveContext<'l, 'tcx>,
        dumper: &'ll mut JsonDumper<O>,
    ) -> DumpVisitor<'l, 'tcx, 'll, O> {
        let span_utils = SpanUtils::new(&save_ctxt.tcx.sess);
        DumpVisitor {
            tcx: save_ctxt.tcx,
            save_ctxt,
            dumper,
            span: span_utils,
            cur_scope: CRATE_NODE_ID,
            // mac_defs: FxHashSet::default(),
            // macro_calls: FxHashSet::default(),
        }
    }

    fn nest_scope<F>(&mut self, scope_id: NodeId, f: F)
    where
        F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, O>),
    {
        let parent_scope = self.cur_scope;
        self.cur_scope = scope_id;
        f(self);
        self.cur_scope = parent_scope;
    }

    fn nest_tables<F>(&mut self, item_id: NodeId, f: F)
    where
        F: FnOnce(&mut DumpVisitor<'l, 'tcx, 'll, O>),
    {
        let item_def_id = self.tcx.hir().local_def_id(item_id);
        if self.tcx.has_typeck_tables(item_def_id) {
            let tables = self.tcx.typeck_tables_of(item_def_id);
            let old_tables = self.save_ctxt.tables;
            self.save_ctxt.tables = tables;
            f(self);
            self.save_ctxt.tables = old_tables;
        } else {
            f(self);
        }
    }

    fn span_from_span(&self, span: Span) -> SpanData {
        self.save_ctxt.span_from_span(span)
    }

    pub fn dump_crate_info(&mut self, name: &str, krate: &ast::Crate) {
        let source_file = self.tcx.sess.local_crate_source_file.as_ref();
        let crate_root = source_file.map(|source_file| {
            let source_file = Path::new(source_file);
            match source_file.file_name() {
                Some(_) => source_file.parent().unwrap().display(),
                None => source_file.display(),
            }.to_string()
        });

        let data = CratePreludeData {
            crate_id: GlobalCrateId {
                name: name.into(),
                disambiguator: self.tcx
                    .sess
                    .local_crate_disambiguator()
                    .to_fingerprint()
                    .as_value(),
            },
            crate_root: crate_root.unwrap_or_else(|| "<no source>".to_owned()),
            external_crates: self.save_ctxt.get_external_crates(),
            span: self.span_from_span(krate.span),
        };

        self.dumper.crate_prelude(data);
    }

    pub fn dump_compilation_options(&mut self, input: &Input, crate_name: &str) {
        // Apply possible `remap-path-prefix` remapping to the input source file
        // (and don't include remapping args anymore)
        let (program, arguments) = {
            let remap_arg_indices = {
                let mut indices = FxHashSet::default();
                // Args are guaranteed to be valid UTF-8 (checked early)
                for (i, e) in env::args().enumerate() {
                    if e.starts_with("--remap-path-prefix=") {
                        indices.insert(i);
                    } else if e == "--remap-path-prefix" {
                        indices.insert(i);
                        indices.insert(i + 1);
                    }
                }
                indices
            };

            let mut args = env::args()
                .enumerate()
                .filter(|(i, _)| !remap_arg_indices.contains(i))
                .map(|(_, arg)| {
                    match input {
                        Input::File(ref path) if path == Path::new(&arg) => {
                            let mapped = &self.tcx.sess.local_crate_source_file;
                            mapped
                                .as_ref()
                                .unwrap()
                                .to_string_lossy()
                                .into()
                        },
                        _ => arg,
                    }
                });

            (args.next().unwrap(), args.collect())
        };

        let data = CompilationOptions {
            directory: self.tcx.sess.working_dir.0.clone(),
            program,
            arguments,
            output: self.save_ctxt.compilation_output(crate_name),
        };

        self.dumper.compilation_opts(data);
    }

    fn write_sub_paths(&mut self, path: &ast::Path) {
        for seg in &path.segments {
            if let Some(data) = self.save_ctxt.get_path_segment_data(seg) {
                self.dumper.dump_ref(data);
            }
        }
    }

    // As write_sub_paths, but does not process the last ident in the path (assuming it
    // will be processed elsewhere). See note on write_sub_paths about global.
    fn write_sub_paths_truncated(&mut self, path: &ast::Path) {
        for seg in &path.segments[..path.segments.len() - 1] {
            if let Some(data) = self.save_ctxt.get_path_segment_data(seg) {
                self.dumper.dump_ref(data);
            }
        }
    }

    fn lookup_def_id(&self, ref_id: NodeId) -> Option<DefId> {
        match self.save_ctxt.get_path_res(ref_id) {
            Res::PrimTy(..) | Res::SelfTy(..) | Res::Err => None,
            def => Some(def.def_id()),
        }
    }

    fn process_formals(&mut self, formals: &'l [ast::Arg], qualname: &str) {
        for arg in formals {
            self.visit_pat(&arg.pat);
            let mut collector = PathCollector::new();
            collector.visit_pat(&arg.pat);

            for (id, ident, ..) in collector.collected_idents {
                let hir_id = self.tcx.hir().node_to_hir_id(id);
                let typ = match self.save_ctxt.tables.node_type_opt(hir_id) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                if !self.span.filter_generated(ident.span) {
                    let id = id_from_node_id(id, &self.save_ctxt);
                    let span = self.span_from_span(ident.span);

                    self.dumper.dump_def(
                        &Access {
                            public: false,
                            reachable: false,
                        },
                        Def {
                            kind: DefKind::Local,
                            id,
                            span,
                            name: ident.to_string(),
                            qualname: format!("{}::{}", qualname, ident.to_string()),
                            value: typ,
                            parent: None,
                            children: vec![],
                            decl_id: None,
                            docs: String::new(),
                            sig: None,
                            attributes: vec![],
                        },
                    );
                }
            }
        }
    }

    fn process_method(
        &mut self,
        sig: &'l ast::MethodSig,
        body: Option<&'l ast::Block>,
        id: ast::NodeId,
        ident: ast::Ident,
        generics: &'l ast::Generics,
        vis: ast::Visibility,
        span: Span,
    ) {
        debug!("process_method: {}:{}", id, ident);

        if let Some(mut method_data) = self.save_ctxt.get_method_data(id, ident, span) {
            let sig_str = crate::make_signature(&sig.decl, &generics);
            if body.is_some() {
                self.nest_tables(
                    id,
                    |v| v.process_formals(&sig.decl.inputs, &method_data.qualname),
                );
            }

            self.process_generic_params(&generics, &method_data.qualname, id);

            method_data.value = sig_str;
            method_data.sig = sig::method_signature(id, ident, generics, sig, &self.save_ctxt);
            let hir_id = self.tcx.hir().node_to_hir_id(id);
            self.dumper.dump_def(&access_from_vis!(self.save_ctxt, vis, hir_id), method_data);
        }

        // walk arg and return types
        for arg in &sig.decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::FunctionRetTy::Ty(ref ret_ty) = sig.decl.output {
            self.visit_ty(ret_ty);
        }

        // walk the fn body
        if let Some(body) = body {
            self.nest_tables(id, |v| v.nest_scope(id, |v| v.visit_block(body)));
        }
    }

    fn process_struct_field_def(&mut self, field: &ast::StructField, parent_id: NodeId) {
        let field_data = self.save_ctxt.get_field_data(field, parent_id);
        if let Some(field_data) = field_data {
            let hir_id = self.tcx.hir().node_to_hir_id(field.id);
            self.dumper.dump_def(&access_from!(self.save_ctxt, field, hir_id), field_data);
        }
    }

    // Dump generic params bindings, then visit_generics
    fn process_generic_params(
        &mut self,
        generics: &'l ast::Generics,
        prefix: &str,
        id: NodeId,
    ) {
        for param in &generics.params {
            match param.kind {
                ast::GenericParamKind::Lifetime { .. } => {}
                ast::GenericParamKind::Type { .. } => {
                    let param_ss = param.ident.span;
                    let name = escape(self.span.snippet(param_ss));
                    // Append $id to name to make sure each one is unique.
                    let qualname = format!("{}::{}${}", prefix, name, id);
                    if !self.span.filter_generated(param_ss) {
                        let id = id_from_node_id(param.id, &self.save_ctxt);
                        let span = self.span_from_span(param_ss);

                        self.dumper.dump_def(
                            &Access {
                                public: false,
                                reachable: false,
                            },
                            Def {
                                kind: DefKind::Type,
                                id,
                                span,
                                name,
                                qualname,
                                value: String::new(),
                                parent: None,
                                children: vec![],
                                decl_id: None,
                                docs: String::new(),
                                sig: None,
                                attributes: vec![],
                            },
                        );
                    }
                }
                ast::GenericParamKind::Const { .. } => {}
            }
        }
        self.visit_generics(generics);
    }

    fn process_fn(
        &mut self,
        item: &'l ast::Item,
        decl: &'l ast::FnDecl,
        ty_params: &'l ast::Generics,
        body: &'l ast::Block,
    ) {
        if let Some(fn_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(fn_data, DefData, item.span);
            self.nest_tables(
                item.id,
                |v| v.process_formals(&decl.inputs, &fn_data.qualname),
            );
            self.process_generic_params(ty_params, &fn_data.qualname, item.id);
            let hir_id = self.tcx.hir().node_to_hir_id(item.id);
            self.dumper.dump_def(&access_from!(self.save_ctxt, item, hir_id), fn_data);
        }

        for arg in &decl.inputs {
            self.visit_ty(&arg.ty);
        }

        if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
            self.visit_ty(&ret_ty);
        }

        self.nest_tables(item.id, |v| v.nest_scope(item.id, |v| v.visit_block(&body)));
    }

    fn process_static_or_const_item(
        &mut self,
        item: &'l ast::Item,
        typ: &'l ast::Ty,
        expr: &'l ast::Expr,
    ) {
        let hir_id = self.tcx.hir().node_to_hir_id(item.id);
        self.nest_tables(item.id, |v| {
            if let Some(var_data) = v.save_ctxt.get_item_data(item) {
                down_cast_data!(var_data, DefData, item.span);
                v.dumper.dump_def(&access_from!(v.save_ctxt, item, hir_id), var_data);
            }
            v.visit_ty(&typ);
            v.visit_expr(expr);
        });
    }

    fn process_assoc_const(
        &mut self,
        id: ast::NodeId,
        ident: ast::Ident,
        typ: &'l ast::Ty,
        expr: Option<&'l ast::Expr>,
        parent_id: DefId,
        vis: ast::Visibility,
        attrs: &'l [Attribute],
    ) {
        let qualname = format!("::{}",
            self.tcx.def_path_str(self.tcx.hir().local_def_id(id)));

        if !self.span.filter_generated(ident.span) {
            let sig = sig::assoc_const_signature(id, ident.name, typ, expr, &self.save_ctxt);
            let span = self.span_from_span(ident.span);
            let hir_id = self.tcx.hir().node_to_hir_id(id);

            self.dumper.dump_def(
                &access_from_vis!(self.save_ctxt, vis, hir_id),
                Def {
                    kind: DefKind::Const,
                    id: id_from_node_id(id, &self.save_ctxt),
                    span,
                    name: ident.name.to_string(),
                    qualname,
                    value: ty_to_string(&typ),
                    parent: Some(id_from_def_id(parent_id)),
                    children: vec![],
                    decl_id: None,
                    docs: self.save_ctxt.docs_for_attrs(attrs),
                    sig,
                    attributes: lower_attributes(attrs.to_owned(), &self.save_ctxt),
                },
            );
        }

        // walk type and init value
        self.nest_tables(id, |v| {
            v.visit_ty(typ);
            if let Some(expr) = expr {
                v.visit_expr(expr);
            }
        });
    }

    // FIXME tuple structs should generate tuple-specific data.
    fn process_struct(
        &mut self,
        item: &'l ast::Item,
        def: &'l ast::VariantData,
        ty_params: &'l ast::Generics,
    ) {
        debug!("process_struct {:?} {:?}", item, item.span);
        let name = item.ident.to_string();
        let qualname = format!("::{}",
            self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));

        let kind = match item.node {
            ast::ItemKind::Struct(_, _) => DefKind::Struct,
            ast::ItemKind::Union(_, _) => DefKind::Union,
            _ => unreachable!(),
        };

        let (value, fields) = match item.node {
            ast::ItemKind::Struct(ast::VariantData::Struct(ref fields, ..), ..) |
            ast::ItemKind::Union(ast::VariantData::Struct(ref fields, ..), ..) => {
                let include_priv_fields = !self.save_ctxt.config.pub_only;
                let fields_str = fields
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        if include_priv_fields || f.vis.node.is_pub() {
                            f.ident
                                .map(|i| i.to_string())
                                .or_else(|| Some(i.to_string()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let value = format!("{} {{ {} }}", name, fields_str);
                (
                    value,
                    fields
                        .iter()
                        .map(|f| id_from_node_id(f.id, &self.save_ctxt))
                        .collect(),
                )
            }
            _ => (String::new(), vec![]),
        };

        if !self.span.filter_generated(item.ident.span) {
            let span = self.span_from_span(item.ident.span);
            let hir_id = self.tcx.hir().node_to_hir_id(item.id);
            self.dumper.dump_def(
                &access_from!(self.save_ctxt, item, hir_id),
                Def {
                    kind,
                    id: id_from_node_id(item.id, &self.save_ctxt),
                    span,
                    name,
                    qualname: qualname.clone(),
                    value,
                    parent: None,
                    children: fields,
                    decl_id: None,
                    docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, &self.save_ctxt),
                    attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
                },
            );
        }

        for field in def.fields() {
            self.process_struct_field_def(field, item.id);
            self.visit_ty(&field.ty);
        }

        self.process_generic_params(ty_params, &qualname, item.id);
    }

    fn process_enum(
        &mut self,
        item: &'l ast::Item,
        enum_definition: &'l ast::EnumDef,
        ty_params: &'l ast::Generics,
    ) {
        let enum_data = self.save_ctxt.get_item_data(item);
        let enum_data = match enum_data {
            None => return,
            Some(data) => data,
        };
        down_cast_data!(enum_data, DefData, item.span);

        let hir_id = self.tcx.hir().node_to_hir_id(item.id);
        let access = access_from!(self.save_ctxt, item, hir_id);

        for variant in &enum_definition.variants {
            let name = variant.node.ident.name.to_string();
            let qualname = format!("{}::{}", enum_data.qualname, name);
            let name_span = variant.node.ident.span;

            match variant.node.data {
                ast::VariantData::Struct(ref fields, ..) => {
                    let fields_str = fields
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            f.ident.map(|i| i.to_string()).unwrap_or_else(|| i.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let value = format!("{}::{} {{ {} }}", enum_data.name, name, fields_str);
                    if !self.span.filter_generated(name_span) {
                        let span = self.span_from_span(name_span);
                        let id = id_from_node_id(variant.node.id, &self.save_ctxt);
                        let parent = Some(id_from_node_id(item.id, &self.save_ctxt));

                        self.dumper.dump_def(
                            &access,
                            Def {
                                kind: DefKind::StructVariant,
                                id,
                                span,
                                name,
                                qualname,
                                value,
                                parent,
                                children: vec![],
                                decl_id: None,
                                docs: self.save_ctxt.docs_for_attrs(&variant.node.attrs),
                                sig: sig::variant_signature(variant, &self.save_ctxt),
                                attributes: lower_attributes(
                                    variant.node.attrs.clone(),
                                    &self.save_ctxt,
                                ),
                            },
                        );
                    }
                }
                ref v => {
                    let mut value = format!("{}::{}", enum_data.name, name);
                    if let &ast::VariantData::Tuple(ref fields, _) = v {
                        value.push('(');
                        value.push_str(&fields
                            .iter()
                            .map(|f| ty_to_string(&f.ty))
                            .collect::<Vec<_>>()
                            .join(", "));
                        value.push(')');
                    }
                    if !self.span.filter_generated(name_span) {
                        let span = self.span_from_span(name_span);
                        let id = id_from_node_id(variant.node.id, &self.save_ctxt);
                        let parent = Some(id_from_node_id(item.id, &self.save_ctxt));

                        self.dumper.dump_def(
                            &access,
                            Def {
                                kind: DefKind::TupleVariant,
                                id,
                                span,
                                name,
                                qualname,
                                value,
                                parent,
                                children: vec![],
                                decl_id: None,
                                docs: self.save_ctxt.docs_for_attrs(&variant.node.attrs),
                                sig: sig::variant_signature(variant, &self.save_ctxt),
                                attributes: lower_attributes(
                                    variant.node.attrs.clone(),
                                    &self.save_ctxt,
                                ),
                            },
                        );
                    }
                }
            }


            for field in variant.node.data.fields() {
                self.process_struct_field_def(field, variant.node.id);
                self.visit_ty(&field.ty);
            }
        }
        self.process_generic_params(ty_params, &enum_data.qualname, item.id);
        self.dumper.dump_def(&access, enum_data);
    }

    fn process_impl(
        &mut self,
        item: &'l ast::Item,
        generics: &'l ast::Generics,
        trait_ref: &'l Option<ast::TraitRef>,
        typ: &'l ast::Ty,
        impl_items: &'l [ast::ImplItem],
    ) {
        if let Some(impl_data) = self.save_ctxt.get_item_data(item) {
            if !self.span.filter_generated(item.span) {
                if let super::Data::RelationData(rel, imp) = impl_data {
                    self.dumper.dump_relation(rel);
                    self.dumper.dump_impl(imp);
                } else {
                    span_bug!(item.span, "unexpected data kind: {:?}", impl_data);
                }
            }
        }
        self.visit_ty(&typ);
        if let &Some(ref trait_ref) = trait_ref {
            self.process_path(trait_ref.ref_id, &trait_ref.path);
        }
        self.process_generic_params(generics, "", item.id);
        for impl_item in impl_items {
            let map = &self.tcx.hir();
            self.process_impl_item(impl_item, map.local_def_id(item.id));
        }
    }

    fn process_trait(
        &mut self,
        item: &'l ast::Item,
        generics: &'l ast::Generics,
        trait_refs: &'l ast::GenericBounds,
        methods: &'l [ast::TraitItem],
    ) {
        let name = item.ident.to_string();
        let qualname = format!("::{}",
            self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
        let mut val = name.clone();
        if !generics.params.is_empty() {
            val.push_str(&generic_params_to_string(&generics.params));
        }
        if !trait_refs.is_empty() {
            val.push_str(": ");
            val.push_str(&bounds_to_string(trait_refs));
        }
        if !self.span.filter_generated(item.ident.span) {
            let id = id_from_node_id(item.id, &self.save_ctxt);
            let span = self.span_from_span(item.ident.span);
            let children = methods
                .iter()
                .map(|i| id_from_node_id(i.id, &self.save_ctxt))
                .collect();
            let hir_id = self.tcx.hir().node_to_hir_id(item.id);
            self.dumper.dump_def(
                &access_from!(self.save_ctxt, item, hir_id),
                Def {
                    kind: DefKind::Trait,
                    id,
                    span,
                    name,
                    qualname: qualname.clone(),
                    value: val,
                    parent: None,
                    children,
                    decl_id: None,
                    docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                    sig: sig::item_signature(item, &self.save_ctxt),
                    attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
                },
            );
        }

        // super-traits
        for super_bound in trait_refs.iter() {
            let trait_ref = match *super_bound {
                ast::GenericBound::Trait(ref trait_ref, _) => trait_ref,
                ast::GenericBound::Outlives(..) => continue,
            };

            let trait_ref = &trait_ref.trait_ref;
            if let Some(id) = self.lookup_def_id(trait_ref.ref_id) {
                let sub_span = trait_ref.path.segments.last().unwrap().ident.span;
                if !self.span.filter_generated(sub_span) {
                    let span = self.span_from_span(sub_span);
                    self.dumper.dump_ref(Ref {
                        kind: RefKind::Type,
                        span: span.clone(),
                        ref_id: id_from_def_id(id),
                    });

                    self.dumper.dump_relation(Relation {
                        kind: RelationKind::SuperTrait,
                        span,
                        from: id_from_def_id(id),
                        to: id_from_node_id(item.id, &self.save_ctxt),
                    });
                }
            }
        }

        // walk generics and methods
        self.process_generic_params(generics, &qualname, item.id);
        for method in methods {
            let map = &self.tcx.hir();
            self.process_trait_item(method, map.local_def_id(item.id))
        }
    }

    // `item` is the module in question, represented as an item.
    fn process_mod(&mut self, item: &ast::Item) {
        if let Some(mod_data) = self.save_ctxt.get_item_data(item) {
            down_cast_data!(mod_data, DefData, item.span);
            let hir_id = self.tcx.hir().node_to_hir_id(item.id);
            self.dumper.dump_def(&access_from!(self.save_ctxt, item, hir_id), mod_data);
        }
    }

    fn dump_path_ref(&mut self, id: NodeId, path: &ast::Path) {
        let path_data = self.save_ctxt.get_path_data(id, path);
        if let Some(path_data) = path_data {
            self.dumper.dump_ref(path_data);
        }
    }

    fn process_path(&mut self, id: NodeId, path: &'l ast::Path) {
        if self.span.filter_generated(path.span) {
            return;
        }
        self.dump_path_ref(id, path);

        // Type arguments
        for seg in &path.segments {
            if let Some(ref generic_args) = seg.args {
                match **generic_args {
                    ast::GenericArgs::AngleBracketed(ref data) => {
                        for arg in &data.args {
                            match arg {
                                ast::GenericArg::Type(ty) => self.visit_ty(ty),
                                _ => {}
                            }
                        }
                    }
                    ast::GenericArgs::Parenthesized(ref data) => {
                        for t in &data.inputs {
                            self.visit_ty(t);
                        }
                        if let Some(ref t) = data.output {
                            self.visit_ty(t);
                        }
                    }
                }
            }
        }

        self.write_sub_paths_truncated(path);
    }

    fn process_struct_lit(
        &mut self,
        ex: &'l ast::Expr,
        path: &'l ast::Path,
        fields: &'l [ast::Field],
        variant: &'l ty::VariantDef,
        base: &'l Option<P<ast::Expr>>,
    ) {
        if let Some(struct_lit_data) = self.save_ctxt.get_expr_data(ex) {
            self.write_sub_paths_truncated(path);
            down_cast_data!(struct_lit_data, RefData, ex.span);
            if !generated_code(ex.span) {
                self.dumper.dump_ref(struct_lit_data);
            }

            for field in fields {
                if let Some(field_data) = self.save_ctxt.get_field_ref_data(field, variant) {
                    self.dumper.dump_ref(field_data);
                }

                self.visit_expr(&field.expr)
            }
        }

        walk_list!(self, visit_expr, base);
    }

    fn process_method_call(
        &mut self,
        ex: &'l ast::Expr,
        seg: &'l ast::PathSegment,
        args: &'l [P<ast::Expr>],
    ) {
        debug!("process_method_call {:?} {:?}", ex, ex.span);
        if let Some(mcd) = self.save_ctxt.get_expr_data(ex) {
            down_cast_data!(mcd, RefData, ex.span);
            if !generated_code(ex.span) {
                self.dumper.dump_ref(mcd);
            }
        }

        // Explicit types in the turbo-fish.
        if let Some(ref generic_args) = seg.args {
            if let ast::GenericArgs::AngleBracketed(ref data) = **generic_args {
                for arg in &data.args {
                    match arg {
                        ast::GenericArg::Type(ty) => self.visit_ty(ty),
                        _ => {}
                    }
                }
            }
        }

        // walk receiver and args
        walk_list!(self, visit_expr, args);
    }

    fn process_pat(&mut self, p: &'l ast::Pat) {
        match p.node {
            PatKind::Struct(ref _path, ref fields, _) => {
                // FIXME do something with _path?
                let hir_id = self.tcx.hir().node_to_hir_id(p.id);
                let adt = match self.save_ctxt.tables.node_type_opt(hir_id) {
                    Some(ty) => ty.ty_adt_def().unwrap(),
                    None => {
                        visit::walk_pat(self, p);
                        return;
                    }
                };
                let variant = adt.variant_of_res(self.save_ctxt.get_path_res(p.id));

                for &Spanned { node: ref field, .. } in fields {
                    if let Some(index) = self.tcx.find_field_index(field.ident, variant) {
                        if !self.span.filter_generated(field.ident.span) {
                            let span = self.span_from_span(field.ident.span);
                            self.dumper.dump_ref(Ref {
                                kind: RefKind::Variable,
                                span,
                                ref_id: id_from_def_id(variant.fields[index].did),
                            });
                        }
                    }
                    self.visit_pat(&field.pat);
                }
            }
            _ => visit::walk_pat(self, p),
        }
    }

    fn process_var_decl_multi(&mut self, pats: &'l [P<ast::Pat>]) {
        let mut collector = PathCollector::new();
        for pattern in pats {
            // collect paths from the arm's patterns
            collector.visit_pat(&pattern);
            self.visit_pat(&pattern);
        }

        // process collected paths
        for (id, ident, immut) in collector.collected_idents {
            match self.save_ctxt.get_path_res(id) {
                Res::Local(hir_id) => {
                    let mut value = if immut == ast::Mutability::Immutable {
                        self.span.snippet(ident.span)
                    } else {
                        "<mutable>".to_owned()
                    };
                    let id = self.tcx.hir().hir_to_node_id(hir_id);
                    let typ = self.save_ctxt
                        .tables
                        .node_type_opt(hir_id)
                        .map(|t| t.to_string())
                        .unwrap_or_default();
                    value.push_str(": ");
                    value.push_str(&typ);

                    if !self.span.filter_generated(ident.span) {
                        let qualname = format!("{}${}", ident.to_string(), id);
                        let id = id_from_node_id(id, &self.save_ctxt);
                        let span = self.span_from_span(ident.span);

                        self.dumper.dump_def(
                            &Access {
                                public: false,
                                reachable: false,
                            },
                            Def {
                                kind: DefKind::Local,
                                id,
                                span,
                                name: ident.to_string(),
                                qualname,
                                value: typ,
                                parent: None,
                                children: vec![],
                                decl_id: None,
                                docs: String::new(),
                                sig: None,
                                attributes: vec![],
                            },
                        );
                    }
                }
                Res::Def(HirDefKind::Ctor(..), _) |
                Res::Def(HirDefKind::Const, _) |
                Res::Def(HirDefKind::AssocConst, _) |
                Res::Def(HirDefKind::Struct, _) |
                Res::Def(HirDefKind::Variant, _) |
                Res::Def(HirDefKind::TyAlias, _) |
                Res::Def(HirDefKind::AssocTy, _) |
                Res::SelfTy(..) => {
                    self.dump_path_ref(id, &ast::Path::from_ident(ident));
                }
                def => error!(
                    "unexpected definition kind when processing collected idents: {:?}",
                    def
                ),
            }
        }

        for (id, ref path) in collector.collected_paths {
            self.process_path(id, path);
        }
    }

    fn process_var_decl(&mut self, p: &'l ast::Pat, value: String) {
        // The local could declare multiple new vars, we must walk the
        // pattern and collect them all.
        let mut collector = PathCollector::new();
        collector.visit_pat(&p);
        self.visit_pat(&p);

        for (id, ident, immut) in collector.collected_idents {
            let mut value = match immut {
                ast::Mutability::Immutable => value.to_string(),
                _ => String::new(),
            };
            let hir_id = self.tcx.hir().node_to_hir_id(id);
            let typ = match self.save_ctxt.tables.node_type_opt(hir_id) {
                Some(typ) => {
                    let typ = typ.to_string();
                    if !value.is_empty() {
                        value.push_str(": ");
                    }
                    value.push_str(&typ);
                    typ
                }
                None => String::new(),
            };

            // Rust uses the id of the pattern for var lookups, so we'll use it too.
            if !self.span.filter_generated(ident.span) {
                let qualname = format!("{}${}", ident.to_string(), id);
                let id = id_from_node_id(id, &self.save_ctxt);
                let span = self.span_from_span(ident.span);

                self.dumper.dump_def(
                    &Access {
                        public: false,
                        reachable: false,
                    },
                    Def {
                        kind: DefKind::Local,
                        id,
                        span,
                        name: ident.to_string(),
                        qualname,
                        value: typ,
                        parent: None,
                        children: vec![],
                        decl_id: None,
                        docs: String::new(),
                        sig: None,
                        attributes: vec![],
                    },
                );
            }
        }
    }

    /// Extracts macro use and definition information from the AST node defined
    /// by the given NodeId, using the expansion information from the node's
    /// span.
    ///
    /// If the span is not macro-generated, do nothing, else use callee and
    /// callsite spans to record macro definition and use data, using the
    /// mac_uses and mac_defs sets to prevent multiples.
    fn process_macro_use(&mut self, _span: Span) {
        // FIXME if we're not dumping the defs (see below), there is no point
        // dumping refs either.
        // let source_span = span.source_callsite();
        // if !self.macro_calls.insert(source_span) {
        //     return;
        // }

        // let data = match self.save_ctxt.get_macro_use_data(span) {
        //     None => return,
        //     Some(data) => data,
        // };

        // self.dumper.macro_use(data);

        // FIXME write the macro def
        // let mut hasher = DefaultHasher::new();
        // data.callee_span.hash(&mut hasher);
        // let hash = hasher.finish();
        // let qualname = format!("{}::{}", data.name, hash);
        // Don't write macro definition for imported macros
        // if !self.mac_defs.contains(&data.callee_span)
        //     && !data.imported {
        //     self.mac_defs.insert(data.callee_span);
        //     if let Some(sub_span) = self.span.span_for_macro_def_name(data.callee_span) {
        //         self.dumper.macro_data(MacroData {
        //             span: sub_span,
        //             name: data.name.clone(),
        //             qualname: qualname.clone(),
        //             // FIXME where do macro docs come from?
        //             docs: String::new(),
        //         }.lower(self.tcx));
        //     }
        // }
    }

    fn process_trait_item(&mut self, trait_item: &'l ast::TraitItem, trait_id: DefId) {
        self.process_macro_use(trait_item.span);
        let vis_span = trait_item.span.shrink_to_lo();
        match trait_item.node {
            ast::TraitItemKind::Const(ref ty, ref expr) => {
                self.process_assoc_const(
                    trait_item.id,
                    trait_item.ident,
                    &ty,
                    expr.as_ref().map(|e| &**e),
                    trait_id,
                    respan(vis_span, ast::VisibilityKind::Public),
                    &trait_item.attrs,
                );
            }
            ast::TraitItemKind::Method(ref sig, ref body) => {
                self.process_method(
                    sig,
                    body.as_ref().map(|x| &**x),
                    trait_item.id,
                    trait_item.ident,
                    &trait_item.generics,
                    respan(vis_span, ast::VisibilityKind::Public),
                    trait_item.span,
                );
            }
            ast::TraitItemKind::Type(ref bounds, ref default_ty) => {
                // FIXME do something with _bounds (for type refs)
                let name = trait_item.ident.name.to_string();
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(trait_item.id)));

                if !self.span.filter_generated(trait_item.ident.span) {
                    let span = self.span_from_span(trait_item.ident.span);
                    let id = id_from_node_id(trait_item.id, &self.save_ctxt);

                    self.dumper.dump_def(
                        &Access {
                            public: true,
                            reachable: true,
                        },
                        Def {
                            kind: DefKind::Type,
                            id,
                            span,
                            name,
                            qualname,
                            value: self.span.snippet(trait_item.span),
                            parent: Some(id_from_def_id(trait_id)),
                            children: vec![],
                            decl_id: None,
                            docs: self.save_ctxt.docs_for_attrs(&trait_item.attrs),
                            sig: sig::assoc_type_signature(
                                trait_item.id,
                                trait_item.ident,
                                Some(bounds),
                                default_ty.as_ref().map(|ty| &**ty),
                                &self.save_ctxt,
                            ),
                            attributes: lower_attributes(trait_item.attrs.clone(), &self.save_ctxt),
                        },
                    );
                }

                if let &Some(ref default_ty) = default_ty {
                    self.visit_ty(default_ty)
                }
            }
            ast::TraitItemKind::Macro(_) => {}
        }
    }

    fn process_impl_item(&mut self, impl_item: &'l ast::ImplItem, impl_id: DefId) {
        self.process_macro_use(impl_item.span);
        match impl_item.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.process_assoc_const(
                    impl_item.id,
                    impl_item.ident,
                    &ty,
                    Some(expr),
                    impl_id,
                    impl_item.vis.clone(),
                    &impl_item.attrs,
                );
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.process_method(
                    sig,
                    Some(body),
                    impl_item.id,
                    impl_item.ident,
                    &impl_item.generics,
                    impl_item.vis.clone(),
                    impl_item.span,
                );
            }
            ast::ImplItemKind::Type(ref ty) => {
                // FIXME: uses of the assoc type should ideally point to this
                // 'def' and the name here should be a ref to the def in the
                // trait.
                self.visit_ty(ty)
            }
            ast::ImplItemKind::Existential(ref bounds) => {
                // FIXME: uses of the assoc type should ideally point to this
                // 'def' and the name here should be a ref to the def in the
                // trait.
                for bound in bounds.iter() {
                    if let ast::GenericBound::Trait(trait_ref, _) = bound {
                        self.process_path(trait_ref.trait_ref.ref_id, &trait_ref.trait_ref.path)
                    }
                }
            }
            ast::ImplItemKind::Macro(_) => {}
        }
    }

    /// Dumps imports in a use tree recursively.
    ///
    /// A use tree is an import that may contain nested braces (RFC 2128). The `use_tree` parameter
    /// is the current use tree under scrutiny, while `id` and `prefix` are its corresponding node
    /// ID and path. `root_item` is the topmost use tree in the hierarchy.
    ///
    /// If `use_tree` is a simple or glob import, it is dumped into the analysis data. Otherwise,
    /// each child use tree is dumped recursively.
    fn process_use_tree(&mut self,
                         use_tree: &'l ast::UseTree,
                         id: NodeId,
                         root_item: &'l ast::Item,
                         prefix: &ast::Path) {
        let path = &use_tree.prefix;

        // The access is calculated using the current tree ID, but with the root tree's visibility
        // (since nested trees don't have their own visibility).
        let hir_id = self.tcx.hir().node_to_hir_id(id);
        let access = access_from!(self.save_ctxt, root_item, hir_id);

        // The parent `DefId` of a given use tree is always the enclosing item.
        let parent = self.save_ctxt.tcx.hir().opt_local_def_id(id)
            .and_then(|id| self.save_ctxt.tcx.parent(id))
            .map(id_from_def_id);

        match use_tree.kind {
            ast::UseTreeKind::Simple(alias, ..) => {
                let ident = use_tree.ident();
                let path = ast::Path {
                    segments: prefix.segments
                        .iter()
                        .chain(path.segments.iter())
                        .cloned()
                        .collect(),
                    span: path.span,
                };

                let sub_span = path.segments.last().unwrap().ident.span;
                if !self.span.filter_generated(sub_span) {
                    let ref_id = self.lookup_def_id(id).map(|id| id_from_def_id(id));
                    let alias_span = alias.map(|i| self.span_from_span(i.span));
                    let span = self.span_from_span(sub_span);
                    self.dumper.import(&access, Import {
                        kind: ImportKind::Use,
                        ref_id,
                        span,
                        alias_span,
                        name: ident.to_string(),
                        value: String::new(),
                        parent,
                    });
                    self.write_sub_paths_truncated(&path);
                }
            }
            ast::UseTreeKind::Glob => {
                let path = ast::Path {
                    segments: prefix.segments
                        .iter()
                        .chain(path.segments.iter())
                        .cloned()
                        .collect(),
                    span: path.span,
                };

                // Make a comma-separated list of names of imported modules.
                let def_id = self.tcx.hir().local_def_id(id);
                let names = self.tcx.names_imported_by_glob_use(def_id);
                let names: Vec<_> = names.iter().map(|n| n.to_string()).collect();

                // Otherwise it's a span with wrong macro expansion info, which
                // we don't want to track anyway, since it's probably macro-internal `use`
                if let Some(sub_span) =
                    self.span.sub_span_of_token(use_tree.span, token::BinOp(token::Star))
                {
                    if !self.span.filter_generated(use_tree.span) {
                        let span = self.span_from_span(sub_span);

                        self.dumper.import(&access, Import {
                            kind: ImportKind::GlobUse,
                            ref_id: None,
                            span,
                            alias_span: None,
                            name: "*".to_owned(),
                            value: names.join(", "),
                            parent,
                        });
                        self.write_sub_paths(&path);
                    }
                }
            }
            ast::UseTreeKind::Nested(ref nested_items) => {
                let prefix = ast::Path {
                    segments: prefix.segments
                        .iter()
                        .chain(path.segments.iter())
                        .cloned()
                        .collect(),
                    span: path.span,
                };
                for &(ref tree, id) in nested_items {
                    self.process_use_tree(tree, id, root_item, &prefix);
                }
            }
        }
    }

    fn process_bounds(&mut self, bounds: &'l ast::GenericBounds) {
        for bound in bounds {
            if let ast::GenericBound::Trait(ref trait_ref, _) = *bound {
                self.process_path(trait_ref.trait_ref.ref_id, &trait_ref.trait_ref.path)
            }
        }
    }
}

impl<'l, 'tcx, 'll, O: DumpOutput + 'll> Visitor<'l> for DumpVisitor<'l, 'tcx, 'll, O> {
    fn visit_mod(&mut self, m: &'l ast::Mod, span: Span, attrs: &[ast::Attribute], id: NodeId) {
        // Since we handle explicit modules ourselves in visit_item, this should
        // only get called for the root module of a crate.
        assert_eq!(id, ast::CRATE_NODE_ID);

        let qualname = format!("::{}",
            self.tcx.def_path_str(self.tcx.hir().local_def_id(id)));

        let cm = self.tcx.sess.source_map();
        let filename = cm.span_to_filename(span);
        let data_id = id_from_node_id(id, &self.save_ctxt);
        let children = m.items
            .iter()
            .map(|i| id_from_node_id(i.id, &self.save_ctxt))
            .collect();
        let span = self.span_from_span(span);

        self.dumper.dump_def(
            &Access {
                public: true,
                reachable: true,
            },
            Def {
                kind: DefKind::Mod,
                id: data_id,
                name: String::new(),
                qualname,
                span,
                value: filename.to_string(),
                children,
                parent: None,
                decl_id: None,
                docs: self.save_ctxt.docs_for_attrs(attrs),
                sig: None,
                attributes: lower_attributes(attrs.to_owned(), &self.save_ctxt),
            },
        );
        self.nest_scope(id, |v| visit::walk_mod(v, m));
    }

    fn visit_item(&mut self, item: &'l ast::Item) {
        use syntax::ast::ItemKind::*;
        self.process_macro_use(item.span);
        match item.node {
            Use(ref use_tree) => {
                let prefix = ast::Path {
                    segments: vec![],
                    span: DUMMY_SP,
                };
                self.process_use_tree(use_tree, item.id, item, &prefix);
            }
            ExternCrate(_) => {
                let name_span = item.ident.span;
                if !self.span.filter_generated(name_span) {
                    let span = self.span_from_span(name_span);
                    let parent = self.save_ctxt.tcx.hir().opt_local_def_id(item.id)
                        .and_then(|id| self.save_ctxt.tcx.parent(id))
                        .map(id_from_def_id);
                    self.dumper.import(
                        &Access {
                            public: false,
                            reachable: false,
                        },
                        Import {
                            kind: ImportKind::ExternCrate,
                            ref_id: None,
                            span,
                            alias_span: None,
                            name: item.ident.to_string(),
                            value: String::new(),
                            parent,
                        },
                    );
                }
            }
            Fn(ref decl, .., ref ty_params, ref body) => {
                self.process_fn(item, &decl, ty_params, &body)
            }
            Static(ref typ, _, ref expr) => self.process_static_or_const_item(item, typ, expr),
            Const(ref typ, ref expr) => self.process_static_or_const_item(item, &typ, &expr),
            Struct(ref def, ref ty_params) | Union(ref def, ref ty_params) => {
                self.process_struct(item, def, ty_params)
            }
            Enum(ref def, ref ty_params) => self.process_enum(item, def, ty_params),
            Impl(.., ref ty_params, ref trait_ref, ref typ, ref impl_items) => {
                self.process_impl(item, ty_params, trait_ref, &typ, impl_items)
            }
            Trait(_, _, ref generics, ref trait_refs, ref methods) => {
                self.process_trait(item, generics, trait_refs, methods)
            }
            Mod(ref m) => {
                self.process_mod(item);
                self.nest_scope(item.id, |v| visit::walk_mod(v, m));
            }
            Ty(ref ty, ref ty_params) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
                let value = ty_to_string(&ty);
                if !self.span.filter_generated(item.ident.span) {
                    let span = self.span_from_span(item.ident.span);
                    let id = id_from_node_id(item.id, &self.save_ctxt);
                    let hir_id = self.tcx.hir().node_to_hir_id(item.id);

                    self.dumper.dump_def(
                        &access_from!(self.save_ctxt, item, hir_id),
                        Def {
                            kind: DefKind::Type,
                            id,
                            span,
                            name: item.ident.to_string(),
                            qualname: qualname.clone(),
                            value,
                            parent: None,
                            children: vec![],
                            decl_id: None,
                            docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                            sig: sig::item_signature(item, &self.save_ctxt),
                            attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
                        },
                    );
                }

                self.visit_ty(&ty);
                self.process_generic_params(ty_params, &qualname, item.id);
            }
            Existential(ref _bounds, ref ty_params) => {
                let qualname = format!("::{}",
                    self.tcx.def_path_str(self.tcx.hir().local_def_id(item.id)));
                // FIXME do something with _bounds
                let value = String::new();
                if !self.span.filter_generated(item.ident.span) {
                    let span = self.span_from_span(item.ident.span);
                    let id = id_from_node_id(item.id, &self.save_ctxt);
                    let hir_id = self.tcx.hir().node_to_hir_id(item.id);

                    self.dumper.dump_def(
                        &access_from!(self.save_ctxt, item, hir_id),
                        Def {
                            kind: DefKind::Type,
                            id,
                            span,
                            name: item.ident.to_string(),
                            qualname: qualname.clone(),
                            value,
                            parent: None,
                            children: vec![],
                            decl_id: None,
                            docs: self.save_ctxt.docs_for_attrs(&item.attrs),
                            sig: sig::item_signature(item, &self.save_ctxt),
                            attributes: lower_attributes(item.attrs.clone(), &self.save_ctxt),
                        },
                    );
                }

                self.process_generic_params(ty_params, &qualname, item.id);
            }
            Mac(_) => (),
            _ => visit::walk_item(self, item),
        }
    }

    fn visit_generics(&mut self, generics: &'l ast::Generics) {
        for param in &generics.params {
            match param.kind {
                ast::GenericParamKind::Lifetime { .. } => {}
                ast::GenericParamKind::Type { ref default, .. } => {
                    self.process_bounds(&param.bounds);
                    if let Some(ref ty) = default {
                        self.visit_ty(&ty);
                    }
                }
                ast::GenericParamKind::Const { ref ty } => {
                    self.process_bounds(&param.bounds);
                    self.visit_ty(&ty);
                }
            }
        }
        for pred in &generics.where_clause.predicates {
            if let ast::WherePredicate::BoundPredicate(ref wbp) = *pred {
                self.process_bounds(&wbp.bounds);
                self.visit_ty(&wbp.bounded_ty);
            }
        }
    }

    fn visit_ty(&mut self, t: &'l ast::Ty) {
        self.process_macro_use(t.span);
        match t.node {
            ast::TyKind::Path(_, ref path) => {
                if generated_code(t.span) {
                    return;
                }

                if let Some(id) = self.lookup_def_id(t.id) {
                    let sub_span = path.segments.last().unwrap().ident.span;
                    let span = self.span_from_span(sub_span);
                    self.dumper.dump_ref(Ref {
                        kind: RefKind::Type,
                        span,
                        ref_id: id_from_def_id(id),
                    });
                }

                self.write_sub_paths_truncated(path);
                visit::walk_path(self, path);
            }
            ast::TyKind::Array(ref element, ref length) => {
                self.visit_ty(element);
                self.nest_tables(length.id, |v| v.visit_expr(&length.value));
            }
            _ => visit::walk_ty(self, t),
        }
    }

    fn visit_expr(&mut self, ex: &'l ast::Expr) {
        debug!("visit_expr {:?}", ex.node);
        self.process_macro_use(ex.span);
        match ex.node {
            ast::ExprKind::Struct(ref path, ref fields, ref base) => {
                let expr_hir_id = self.save_ctxt.tcx.hir().node_to_hir_id(ex.id);
                let hir_expr = self.save_ctxt.tcx.hir().expect_expr(expr_hir_id);
                let adt = match self.save_ctxt.tables.expr_ty_opt(&hir_expr) {
                    Some(ty) if ty.ty_adt_def().is_some() => ty.ty_adt_def().unwrap(),
                    _ => {
                        visit::walk_expr(self, ex);
                        return;
                    }
                };
                let node_id = self.save_ctxt.tcx.hir().hir_to_node_id(hir_expr.hir_id);
                let res = self.save_ctxt.get_path_res(node_id);
                self.process_struct_lit(ex, path, fields, adt.variant_of_res(res), base)
            }
            ast::ExprKind::MethodCall(ref seg, ref args) => self.process_method_call(ex, seg, args),
            ast::ExprKind::Field(ref sub_ex, _) => {
                self.visit_expr(&sub_ex);

                if let Some(field_data) = self.save_ctxt.get_expr_data(ex) {
                    down_cast_data!(field_data, RefData, ex.span);
                    if !generated_code(ex.span) {
                        self.dumper.dump_ref(field_data);
                    }
                }
            }
            ast::ExprKind::Closure(_, _, _, ref decl, ref body, _fn_decl_span) => {
                let id = format!("${}", ex.id);

                // walk arg and return types
                for arg in &decl.inputs {
                    self.visit_ty(&arg.ty);
                }

                if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
                    self.visit_ty(&ret_ty);
                }

                // walk the body
                self.nest_tables(ex.id, |v| {
                    v.process_formals(&decl.inputs, &id);
                    v.nest_scope(ex.id, |v| v.visit_expr(body))
                });
            }
            ast::ExprKind::ForLoop(ref pattern, ref subexpression, ref block, _) => {
                let value = self.span.snippet(subexpression.span);
                self.process_var_decl(pattern, value);
                debug!("for loop, walk sub-expr: {:?}", subexpression.node);
                self.visit_expr(subexpression);
                visit::walk_block(self, block);
            }
            ast::ExprKind::Let(ref pats, ref scrutinee) => {
                self.process_var_decl_multi(pats);
                self.visit_expr(scrutinee);
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.visit_expr(element);
                self.nest_tables(count.id, |v| v.visit_expr(&count.value));
            }
            // In particular, we take this branch for call and path expressions,
            // where we'll index the idents involved just by continuing to walk.
            _ => visit::walk_expr(self, ex),
        }
    }

    fn visit_mac(&mut self, mac: &'l ast::Mac) {
        // These shouldn't exist in the AST at this point, log a span bug.
        span_bug!(
            mac.span,
            "macro invocation should have been expanded out of AST"
        );
    }

    fn visit_pat(&mut self, p: &'l ast::Pat) {
        self.process_macro_use(p.span);
        self.process_pat(p);
    }

    fn visit_arm(&mut self, arm: &'l ast::Arm) {
        self.process_var_decl_multi(&arm.pats);
        if let Some(expr) = &arm.guard {
            self.visit_expr(expr);
        }
        self.visit_expr(&arm.body);
    }

    fn visit_path(&mut self, p: &'l ast::Path, id: NodeId) {
        self.process_path(id, p);
    }

    fn visit_stmt(&mut self, s: &'l ast::Stmt) {
        self.process_macro_use(s.span);
        visit::walk_stmt(self, s)
    }

    fn visit_local(&mut self, l: &'l ast::Local) {
        self.process_macro_use(l.span);
        let value = l.init
            .as_ref()
            .map(|i| self.span.snippet(i.span))
            .unwrap_or_default();
        self.process_var_decl(&l.pat, value);

        // Just walk the initialiser and type (don't want to walk the pattern again).
        walk_list!(self, visit_ty, &l.ty);
        walk_list!(self, visit_expr, &l.init);
    }

    fn visit_foreign_item(&mut self, item: &'l ast::ForeignItem) {
        let hir_id = self.tcx.hir().node_to_hir_id(item.id);
        let access = access_from!(self.save_ctxt, item, hir_id);

        match item.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                if let Some(fn_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(fn_data, DefData, item.span);

                    self.process_generic_params(generics, &fn_data.qualname, item.id);
                    self.dumper.dump_def(&access, fn_data);
                }

                for arg in &decl.inputs {
                    self.visit_ty(&arg.ty);
                }

                if let ast::FunctionRetTy::Ty(ref ret_ty) = decl.output {
                    self.visit_ty(&ret_ty);
                }
            }
            ast::ForeignItemKind::Static(ref ty, _) => {
                if let Some(var_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(var_data, DefData, item.span);
                    self.dumper.dump_def(&access, var_data);
                }

                self.visit_ty(ty);
            }
            ast::ForeignItemKind::Ty => {
                if let Some(var_data) = self.save_ctxt.get_extern_item_data(item) {
                    down_cast_data!(var_data, DefData, item.span);
                    self.dumper.dump_def(&access, var_data);
                }
            }
            ast::ForeignItemKind::Macro(..) => {}
        }
    }
}
