//! Completion for extern crates

use hir::Name;
use ide_db::{SymbolKind, documentation::HasDocs};
use syntax::ToSmolStr;

use crate::{CompletionItem, CompletionItemKind, context::CompletionContext};

use super::Completions;

pub(crate) fn complete_extern_crate(acc: &mut Completions, ctx: &CompletionContext<'_>) {
    let imported_extern_crates: Vec<Name> = ctx.scope.extern_crate_decls().collect();

    for (name, module) in ctx.scope.extern_crates() {
        if imported_extern_crates.contains(&name) {
            continue;
        }

        let mut item = CompletionItem::new(
            CompletionItemKind::SymbolKind(SymbolKind::Module),
            ctx.source_range(),
            name.display_no_db(ctx.edition).to_smolstr(),
            ctx.edition,
        );
        item.set_documentation(module.docs(ctx.db));

        item.add_to(acc, ctx.db);
    }
}

#[cfg(test)]
mod test {
    use crate::tests::completion_list_no_kw;

    #[test]
    fn can_complete_extern_crate() {
        let case = r#"
//- /lib.rs crate:other_crate_a
// nothing here
//- /other_crate_b.rs crate:other_crate_b
pub mod good_mod{}
//- /lib.rs crate:crate_c
// nothing here
//- /lib.rs crate:lib deps:other_crate_a,other_crate_b,crate_c extern-prelude:other_crate_a
extern crate oth$0
mod other_mod {}
"#;

        let completion_list = completion_list_no_kw(case);

        assert_eq!("md other_crate_a\n".to_owned(), completion_list);
    }

    #[test]
    fn will_not_complete_existing_import() {
        let case = r#"
//- /lib.rs crate:other_crate_a
// nothing here
//- /lib.rs crate:crate_c
// nothing here
//- /lib.rs crate:other_crate_b
//
//- /lib.rs crate:lib deps:other_crate_a,other_crate_b,crate_c extern-prelude:other_crate_a,other_crate_b
extern crate other_crate_b;
extern crate oth$0
mod other_mod {}
"#;

        let completion_list = completion_list_no_kw(case);

        assert_eq!("md other_crate_a\n".to_owned(), completion_list);
    }
}
