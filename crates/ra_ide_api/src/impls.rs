use ra_db::SourceDatabase;
use ra_syntax::{
    AstNode, ast,
    algo::find_node_at_offset,
};
use hir::{db::HirDatabase, source_binder};

use crate::{FilePosition, NavigationTarget, db::RootDatabase, RangeInfo};

pub(crate) fn goto_implementation(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let file = db.parse(position.file_id);
    let syntax = file.syntax();

    let module = source_binder::module_from_position(db, position)?;

    if let Some(nominal_def) = find_node_at_offset::<ast::NominalDef>(syntax, position.offset) {
        return Some(RangeInfo::new(
            nominal_def.syntax().range(),
            impls_for_def(db, nominal_def, module)?,
        ));
    } else if let Some(trait_def) = find_node_at_offset::<ast::TraitDef>(syntax, position.offset) {
        return Some(RangeInfo::new(
            trait_def.syntax().range(),
            impls_for_trait(db, trait_def, module)?,
        ));
    }

    None
}

fn impls_for_def(
    db: &RootDatabase,
    node: &ast::NominalDef,
    module: hir::Module,
) -> Option<Vec<NavigationTarget>> {
    let ty = match node.kind() {
        ast::NominalDefKind::StructDef(def) => {
            source_binder::struct_from_module(db, module, &def).ty(db)
        }
        ast::NominalDefKind::EnumDef(def) => {
            source_binder::enum_from_module(db, module, &def).ty(db)
        }
    };

    let krate = module.krate(db)?;
    let impls = db.impls_in_crate(krate);

    Some(
        impls
            .lookup_impl_blocks(db, &ty)
            .map(|(module, imp)| NavigationTarget::from_impl_block(db, module, &imp))
            .collect(),
    )
}

fn impls_for_trait(
    db: &RootDatabase,
    node: &ast::TraitDef,
    module: hir::Module,
) -> Option<Vec<NavigationTarget>> {
    let tr = source_binder::trait_from_module(db, module, node);

    let krate = module.krate(db)?;
    let impls = db.impls_in_crate(krate);

    Some(
        impls
            .lookup_impl_blocks_for_trait(db, &tr)
            .map(|(module, imp)| NavigationTarget::from_impl_block(db, module, &imp))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixture: &str, expected: &[&str]) {
        let (analysis, pos) = analysis_and_position(fixture);

        let navs = analysis.goto_implementation(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), expected.len());
        navs.into_iter().enumerate().for_each(|(i, nav)| nav.assert_match(expected[i]));
    }

    #[test]
    fn goto_implementation_works() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|>;
            impl Foo {}
            ",
            &["impl IMPL_BLOCK FileId(1) [12; 23)"],
        );
    }

    #[test]
    fn goto_implementation_works_multiple_blocks() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|>;
            impl Foo {}
            impl Foo {}
            ",
            &["impl IMPL_BLOCK FileId(1) [12; 23)", "impl IMPL_BLOCK FileId(1) [24; 35)"],
        );
    }

    #[test]
    fn goto_implementation_works_multiple_mods() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|>;
            mod a {
                impl super::Foo {}
            }
            mod b {
                impl super::Foo {}
            }
            ",
            &["impl IMPL_BLOCK FileId(1) [24; 42)", "impl IMPL_BLOCK FileId(1) [57; 75)"],
        );
    }

    #[test]
    fn goto_implementation_works_multiple_files() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|>;
            mod a;
            mod b;
            //- /a.rs
            impl crate::Foo {}
            //- /b.rs
            impl crate::Foo {}
            ",
            &["impl IMPL_BLOCK FileId(2) [0; 18)", "impl IMPL_BLOCK FileId(3) [0; 18)"],
        );
    }

    #[test]
    fn goto_implementation_for_trait() {
        check_goto(
            "
            //- /lib.rs
            trait T<|> {}
            struct Foo;
            impl T for Foo {}
            ",
            &["impl IMPL_BLOCK FileId(1) [23; 40)"],
        );
    }

    #[test]
    fn goto_implementation_for_trait_multiple_files() {
        check_goto(
            "
            //- /lib.rs
            trait T<|> {};
            struct Foo;
            mod a;
            mod b;
            //- /a.rs
            impl crate::T for crate::Foo {}
            //- /b.rs
            impl crate::T for crate::Foo {}
            ",
            &["impl IMPL_BLOCK FileId(2) [0; 31)", "impl IMPL_BLOCK FileId(3) [0; 31)"],
        );
    }
}
