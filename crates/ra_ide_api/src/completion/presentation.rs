//! This modules takes care of rendering various defenitions as completion items.
use hir::{Docs, HasSource, HirDisplay, PerNs, Resolution, Ty};
use join_to_string::join;
use ra_syntax::ast::NameOwner;
use test_utils::tested_by;

use crate::completion::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

use crate::display::{const_label, function_label, type_label};

impl Completions {
    pub(crate) fn add_field(
        &mut self,
        ctx: &CompletionContext,
        field: hir::StructField,
        substs: &hir::Substs,
    ) {
        CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            field.name(ctx.db).to_string(),
        )
        .kind(CompletionItemKind::Field)
        .detail(field.ty(ctx.db).subst(substs).display(ctx.db).to_string())
        .set_documentation(field.docs(ctx.db))
        .add_to(self);
    }

    pub(crate) fn add_pos_field(&mut self, ctx: &CompletionContext, field: usize, ty: &hir::Ty) {
        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), field.to_string())
            .kind(CompletionItemKind::Field)
            .detail(ty.display(ctx.db).to_string())
            .add_to(self);
    }

    pub(crate) fn add_resolution(
        &mut self,
        ctx: &CompletionContext,
        local_name: String,
        resolution: &PerNs<Resolution>,
    ) {
        use hir::ModuleDef::*;

        let def = resolution.as_ref().take_types().or_else(|| resolution.as_ref().take_values());
        let def = match def {
            None => {
                self.add(CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    local_name,
                ));
                return;
            }
            Some(it) => it,
        };
        let mut completion_kind = CompletionKind::Reference;
        let (kind, docs) = match def {
            Resolution::Def(Module(it)) => (CompletionItemKind::Module, it.docs(ctx.db)),
            Resolution::Def(Function(func)) => {
                return self.add_function_with_name(ctx, Some(local_name), *func);
            }
            Resolution::Def(Struct(it)) => (CompletionItemKind::Struct, it.docs(ctx.db)),
            Resolution::Def(Union(it)) => (CompletionItemKind::Struct, it.docs(ctx.db)),
            Resolution::Def(Enum(it)) => (CompletionItemKind::Enum, it.docs(ctx.db)),
            Resolution::Def(EnumVariant(it)) => (CompletionItemKind::EnumVariant, it.docs(ctx.db)),
            Resolution::Def(Const(it)) => (CompletionItemKind::Const, it.docs(ctx.db)),
            Resolution::Def(Static(it)) => (CompletionItemKind::Static, it.docs(ctx.db)),
            Resolution::Def(Trait(it)) => (CompletionItemKind::Trait, it.docs(ctx.db)),
            Resolution::Def(TypeAlias(it)) => (CompletionItemKind::TypeAlias, it.docs(ctx.db)),
            Resolution::Def(BuiltinType(..)) => {
                completion_kind = CompletionKind::BuiltinType;
                (CompletionItemKind::BuiltinType, None)
            }
            Resolution::GenericParam(..) => (CompletionItemKind::TypeParam, None),
            Resolution::LocalBinding(..) => (CompletionItemKind::Binding, None),
            Resolution::SelfType(..) => (
                CompletionItemKind::TypeParam, // (does this need its own kind?)
                None,
            ),
        };

        let mut completion_item =
            CompletionItem::new(completion_kind, ctx.source_range(), local_name);
        if let Resolution::LocalBinding(pat_id) = def {
            let ty = ctx
                .analyzer
                .type_of_pat_by_id(ctx.db, pat_id.clone())
                .filter(|t| t != &Ty::Unknown)
                .map(|t| t.display(ctx.db).to_string());
            completion_item = completion_item.set_detail(ty);
        };
        completion_item.kind(kind).set_documentation(docs).add_to(self)
    }

    pub(crate) fn add_function(&mut self, ctx: &CompletionContext, func: hir::Function) {
        self.add_function_with_name(ctx, None, func)
    }

    fn add_function_with_name(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        func: hir::Function,
    ) {
        let data = func.data(ctx.db);
        let name = name.unwrap_or_else(|| data.name().to_string());
        let ast_node = func.source(ctx.db).ast;
        let detail = function_label(&ast_node);

        let mut builder = CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name)
            .kind(if data.has_self_param() {
                CompletionItemKind::Method
            } else {
                CompletionItemKind::Function
            })
            .set_documentation(func.docs(ctx.db))
            .detail(detail);
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            tested_by!(inserts_parens_for_function_calls);
            let snippet =
                if data.params().is_empty() || data.has_self_param() && data.params().len() == 1 {
                    format!("{}()$0", data.name())
                } else {
                    format!("{}($0)", data.name())
                };
            builder = builder.insert_snippet(snippet);
        }
        self.add(builder)
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, constant: hir::Const) {
        let ast_node = constant.source(ctx.db).ast;
        let name = match ast_node.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = const_label(&ast_node);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::Const)
            .set_documentation(constant.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_type_alias(&mut self, ctx: &CompletionContext, type_alias: hir::TypeAlias) {
        let type_def = type_alias.source(ctx.db).ast;
        let name = match type_def.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = type_label(&type_def);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::TypeAlias)
            .set_documentation(type_alias.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_enum_variant(&mut self, ctx: &CompletionContext, variant: hir::EnumVariant) {
        let name = match variant.name(ctx.db) {
            Some(it) => it,
            None => return,
        };
        let detail_types = variant.fields(ctx.db).into_iter().map(|field| field.ty(ctx.db));
        let detail = join(detail_types.map(|t| t.display(ctx.db).to_string()))
            .separator(", ")
            .surround_with("(", ")")
            .to_string();

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
            .kind(CompletionItemKind::EnumVariant)
            .set_documentation(variant.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::completion::{check_completion, CompletionKind};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn inserts_parens_for_function_calls() {
        covers!(inserts_parens_for_function_calls);
        check_reference_completion(
            "inserts_parens_for_function_calls1",
            r"
            fn no_args() {}
            fn main() { no_<|> }
            ",
        );
        check_reference_completion(
            "inserts_parens_for_function_calls2",
            r"
            fn with_args(x: i32, y: String) {}
            fn main() { with_<|> }
            ",
        );
        check_reference_completion(
            "inserts_parens_for_function_calls3",
            r"
            struct S {}
            impl S {
                fn foo(&self) {}
            }
            fn bar(s: &S) {
                s.f<|>
            }
            ",
        )
    }

    #[test]
    fn dont_render_function_parens_in_use_item() {
        check_reference_completion(
            "dont_render_function_parens_in_use_item",
            "
            //- /lib.rs
            mod m { pub fn foo() {} }
            use crate::m::f<|>;
            ",
        )
    }

    #[test]
    fn dont_render_function_parens_if_already_call() {
        check_reference_completion(
            "dont_render_function_parens_if_already_call",
            "
            //- /lib.rs
            fn frobnicate() {}
            fn main() {
                frob<|>();
            }
            ",
        );
        check_reference_completion(
            "dont_render_function_parens_if_already_call_assoc_fn",
            "
            //- /lib.rs
            struct Foo {}
            impl Foo { fn new() -> Foo {} }
            fn main() {
                Foo::ne<|>();
            }
            ",
        )
    }

}
