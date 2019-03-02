/// This module implements new import-resolution/macro expansion algorithm.
///
/// The result of this module is `CrateDefMap`: a datastructure which contains:
///
///   * a tree of modules for the crate
///   * for each module, a set of items visible in the module (directly declared
///     or imported)
///
/// Note that `CrateDefMap` contains fully macro expanded code.
///
/// Computing `CrateDefMap` can be partitioned into several logically
/// independent "phases". The phases are mutually recursive though, there's no
/// stric ordering.
///
/// ## Collecting RawItems
///
///  This happens in the `raw` module, which parses a single source file into a
///  set of top-level items. Nested importa are desugared to flat imports in
///  this phase. Macro calls are represented as a triple of (Path, Option<Name>,
///  TokenTree).
///
/// ## Collecting Modules
///
/// This happens in the `collector` module. In this phase, we recursively walk
/// tree of modules, collect raw items from submodules, populate module scopes
/// with defined items (so, we assign item ids in this phase) and record the set
/// of unresovled imports and macros.
///
/// While we walk tree of modules, we also record macro_rules defenitions and
/// expand calls to macro_rules defined macros.
///
/// ## Resolving Imports
///
/// TBD
///
/// ## Resolving Macros
///
/// While macro_rules from the same crate use a global mutable namespace, macros
/// from other crates (including proc-macros) can be used with `foo::bar!`
/// syntax.
///
/// TBD;
mod raw;
mod collector;

use rustc_hash::FxHashMap;
use ra_arena::{Arena};

use crate::{
    Name,
    module_tree::ModuleId,
    nameres::ModuleScope,
};

#[derive(Default, Debug)]
struct ModuleData {
    parent: Option<ModuleId>,
    children: FxHashMap<Name, ModuleId>,
    scope: ModuleScope,
}

/// Contans all top-level defs from a macro-expanded crate
#[derive(Debug)]
pub(crate) struct CrateDefMap {
    root: ModuleId,
    modules: Arena<ModuleId, ModuleData>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ra_db::SourceDatabase;
    use insta::assert_snapshot_matches;

    use crate::{Crate, mock::MockDatabase, nameres::Resolution};

    use super::*;

    fn compute_crate_def_map(fixture: &str) -> Arc<CrateDefMap> {
        let db = MockDatabase::with_files(fixture);
        let crate_id = db.crate_graph().iter().next().unwrap();
        let krate = Crate { crate_id };
        collector::crate_def_map_query(&db, krate)
    }

    fn render_crate_def_map(map: &CrateDefMap) -> String {
        let mut buf = String::new();
        go(&mut buf, map, "\ncrate", map.root);
        return buf;

        fn go(buf: &mut String, map: &CrateDefMap, path: &str, module: ModuleId) {
            *buf += path;
            *buf += "\n";
            for (name, res) in map.modules[module].scope.items.iter() {
                *buf += &format!("{}: {}\n", name, dump_resolution(res))
            }
            for (name, child) in map.modules[module].children.iter() {
                let path = path.to_string() + &format!("::{}", name);
                go(buf, map, &path, *child);
            }
        }

        fn dump_resolution(resolution: &Resolution) -> &'static str {
            match (resolution.def.types.is_some(), resolution.def.values.is_some()) {
                (true, true) => "t v",
                (true, false) => "t",
                (false, true) => "v",
                (false, false) => "_",
            }
        }
    }

    fn def_map(fixtute: &str) -> String {
        let dm = compute_crate_def_map(fixtute);
        render_crate_def_map(&dm)
    }

    #[test]
    fn crate_def_map_smoke_test() {
        let map = def_map(
            "
            //- /lib.rs
            mod foo;
            struct S;

            //- /foo/mod.rs
            pub mod bar;
            fn f() {}

            //- /foo/bar.rs
            pub struct Baz;
            enum E { V }
        ",
        );
        assert_snapshot_matches!(
        map,
            @r###"
crate
S: t v

crate::foo
f: v

crate::foo::bar
Baz: t v
E: t
"###
        )
    }

    #[test]
    fn macro_rules_are_globally_visible() {
        let map = def_map(
            "
            //- /lib.rs
            macro_rules! structs {
                ($($i:ident),*) => {
                    $(struct $i { field: u32 } )*
                }
            }
            structs!(Foo);
            mod nested;

            //- /nested.rs
            structs!(Bar, Baz);
        ",
        );
        assert_snapshot_matches!(map, @r###"
crate
Foo: t v

crate::nested
Bar: t v
Baz: t v
"###);
    }

    #[test]
    fn macro_rules_can_define_modules() {
        let map = def_map(
            "
            //- /lib.rs
            macro_rules! m {
                ($name:ident) => { mod $name;  }
            }
            m!(n1);

            //- /n1.rs
            m!(n2)
            //- /n1/n2.rs
            struct X;
        ",
        );
        assert_snapshot_matches!(map, @r###"
crate

crate::n1

crate::n1::n2
X: t v
"###);
    }
}
