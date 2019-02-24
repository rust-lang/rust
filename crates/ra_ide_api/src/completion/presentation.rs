//! This modules takes care of rendering various defenitions as completion items.
use join_to_string::join;
use test_utils::tested_by;
use hir::Docs;

use crate::completion::{
    Completions, CompletionKind, CompletionItemKind, CompletionContext, CompletionItem,
    function_label,
};

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
        .detail(field.ty(ctx.db).subst(substs).to_string())
        .set_documentation(field.docs(ctx.db))
        .add_to(self);
    }

    pub(crate) fn add_pos_field(&mut self, ctx: &CompletionContext, field: usize, ty: &hir::Ty) {
        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), field.to_string())
            .kind(CompletionItemKind::Field)
            .detail(ty.to_string())
            .add_to(self);
    }

    pub(crate) fn add_function(&mut self, ctx: &CompletionContext, func: hir::Function) {
        let sig = func.signature(ctx.db);

        let mut builder = CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            sig.name().to_string(),
        )
        .kind(if sig.has_self_param() {
            CompletionItemKind::Method
        } else {
            CompletionItemKind::Function
        })
        .set_documentation(func.docs(ctx.db))
        .set_detail(function_item_label(ctx, func));
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            tested_by!(inserts_parens_for_function_calls);
            let snippet =
                if sig.params().is_empty() || sig.has_self_param() && sig.params().len() == 1 {
                    format!("{}()$0", sig.name())
                } else {
                    format!("{}($0)", sig.name())
                };
            builder = builder.insert_snippet(snippet);
        }
        self.add(builder)
    }

    pub(crate) fn add_enum_variant(&mut self, ctx: &CompletionContext, variant: hir::EnumVariant) {
        let name = match variant.name(ctx.db) {
            Some(it) => it,
            None => return,
        };
        let detail_types = variant.fields(ctx.db).into_iter().map(|field| field.ty(ctx.db));
        let detail = join(detail_types).separator(", ").surround_with("(", ")").to_string();

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
            .kind(CompletionItemKind::EnumVariant)
            .set_documentation(variant.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }
}

fn function_item_label(ctx: &CompletionContext, function: hir::Function) -> Option<String> {
    let node = function.source(ctx.db).1;
    function_label(&node)
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::completion::{CompletionKind, completion_item::check_completion};

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
