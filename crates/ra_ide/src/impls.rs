//! FIXME: write short doc here

use hir::{FromSource, ImplBlock};
use ra_db::SourceDatabase;
use ra_syntax::{algo::find_node_at_offset, ast, AstNode};

use crate::{db::RootDatabase, display::ToNav, FilePosition, NavigationTarget, RangeInfo};

pub(crate) fn goto_implementation(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let parse = db.parse(position.file_id);
    let syntax = parse.tree().syntax().clone();

    let src = hir::ModuleSource::from_position(db, position);
    let module = hir::Module::from_definition(
        db,
        hir::InFile { file_id: position.file_id.into(), value: src },
    )?;

    if let Some(nominal_def) = find_node_at_offset::<ast::NominalDef>(&syntax, position.offset) {
        return Some(RangeInfo::new(
            nominal_def.syntax().text_range(),
            impls_for_def(db, position, &nominal_def, module)?,
        ));
    } else if let Some(trait_def) = find_node_at_offset::<ast::TraitDef>(&syntax, position.offset) {
        return Some(RangeInfo::new(
            trait_def.syntax().text_range(),
            impls_for_trait(db, position, &trait_def, module)?,
        ));
    }

    None
}

fn impls_for_def(
    db: &RootDatabase,
    position: FilePosition,
    node: &ast::NominalDef,
    module: hir::Module,
) -> Option<Vec<NavigationTarget>> {
    let ty = match node {
        ast::NominalDef::StructDef(def) => {
            let src = hir::InFile { file_id: position.file_id.into(), value: def.clone() };
            hir::Struct::from_source(db, src)?.ty(db)
        }
        ast::NominalDef::EnumDef(def) => {
            let src = hir::InFile { file_id: position.file_id.into(), value: def.clone() };
            hir::Enum::from_source(db, src)?.ty(db)
        }
        ast::NominalDef::UnionDef(def) => {
            let src = hir::InFile { file_id: position.file_id.into(), value: def.clone() };
            hir::Union::from_source(db, src)?.ty(db)
        }
    };

    let krate = module.krate();
    let impls = ImplBlock::all_in_crate(db, krate);

    Some(
        impls
            .into_iter()
            .filter(|impl_block| ty.is_equal_for_find_impls(&impl_block.target_ty(db)))
            .map(|imp| imp.to_nav(db))
            .collect(),
    )
}

fn impls_for_trait(
    db: &RootDatabase,
    position: FilePosition,
    node: &ast::TraitDef,
    module: hir::Module,
) -> Option<Vec<NavigationTarget>> {
    let src = hir::InFile { file_id: position.file_id.into(), value: node.clone() };
    let tr = hir::Trait::from_source(db, src)?;

    let krate = module.krate();
    let impls = ImplBlock::for_trait(db, krate, tr);

    Some(impls.into_iter().map(|imp| imp.to_nav(db)).collect())
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixture: &str, expected: &[&str]) {
        let (analysis, pos) = analysis_and_position(fixture);

        let mut navs = analysis.goto_implementation(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), expected.len());
        navs.sort_by_key(|nav| (nav.file_id(), nav.full_range().start()));
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

    #[test]
    fn goto_implementation_all_impls() {
        check_goto(
            "
            //- /lib.rs
            trait T {}
            struct Foo<|>;
            impl Foo {}
            impl T for Foo {}
            impl T for &Foo {}
            ",
            &[
                "impl IMPL_BLOCK FileId(1) [23; 34)",
                "impl IMPL_BLOCK FileId(1) [35; 52)",
                "impl IMPL_BLOCK FileId(1) [53; 71)",
            ],
        );
    }
}
