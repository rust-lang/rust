//! This diagnostic provides an assist for creating a struct definition from a JSON
//! example.

use hir::{ImportPathConfig, PathResolution, Semantics};
use ide_db::text_edit::TextEdit;
use ide_db::{
    EditionedFileId, FileRange, FxHashMap, RootDatabase,
    helpers::mod_path_to_ast,
    imports::insert_use::{ImportScope, insert_use},
    source_change::SourceChangeBuilder,
};
use itertools::Itertools;
use stdx::{format_to, never};
use syntax::{
    Edition, SyntaxKind, SyntaxNode,
    ast::{self, make},
};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsConfig, Severity, fix};

#[derive(Default)]
struct State {
    result: String,
    has_serialize: bool,
    has_deserialize: bool,
    names: FxHashMap<String, usize>,
}

impl State {
    fn generate_new_name(&mut self, name: &str) -> ast::Name {
        let name = stdx::to_camel_case(name);
        let count = if let Some(count) = self.names.get_mut(&name) {
            *count += 1;
            *count
        } else {
            self.names.insert(name.clone(), 1);
            1
        };
        make::name(&format!("{name}{count}"))
    }

    fn serde_derive(&self) -> String {
        let mut v = vec![];
        if self.has_serialize {
            v.push("Serialize");
        }
        if self.has_deserialize {
            v.push("Deserialize");
        }
        match v.as_slice() {
            [] => "".to_owned(),
            [x] => format!("#[derive({x})]\n"),
            [x, y] => format!("#[derive({x}, {y})]\n"),
            _ => {
                never!();
                "".to_owned()
            }
        }
    }

    fn build_struct(
        &mut self,
        name: &str,
        value: &serde_json::Map<String, serde_json::Value>,
    ) -> ast::Type {
        let name = self.generate_new_name(name);
        let ty = make::ty(&name.to_string());
        let strukt = make::struct_(
            None,
            name,
            None,
            make::record_field_list(value.iter().sorted_unstable_by_key(|x| x.0).map(
                |(name, value)| {
                    make::record_field(None, make::name(name), self.type_of(name, value))
                },
            ))
            .into(),
        );
        format_to!(self.result, "{}{}\n", self.serde_derive(), strukt);
        ty
    }

    fn type_of(&mut self, name: &str, value: &serde_json::Value) -> ast::Type {
        match value {
            serde_json::Value::Null => make::ty_unit(),
            serde_json::Value::Bool(_) => make::ty("bool"),
            serde_json::Value::Number(it) => make::ty(if it.is_i64() { "i64" } else { "f64" }),
            serde_json::Value::String(_) => make::ty("String"),
            serde_json::Value::Array(it) => {
                let ty = match it.iter().next() {
                    Some(x) => self.type_of(name, x),
                    None => make::ty_placeholder(),
                };
                make::ty(&format!("Vec<{ty}>"))
            }
            serde_json::Value::Object(x) => self.build_struct(name, x),
        }
    }
}

pub(crate) fn json_in_items(
    sema: &Semantics<'_, RootDatabase>,
    acc: &mut Vec<Diagnostic>,
    file_id: EditionedFileId,
    node: &SyntaxNode,
    config: &DiagnosticsConfig,
    edition: Edition,
) {
    (|| {
        if node.kind() == SyntaxKind::ERROR
            && node.first_token().map(|x| x.kind()) == Some(SyntaxKind::L_CURLY)
            && node.last_token().map(|x| x.kind()) == Some(SyntaxKind::R_CURLY)
        {
            let node_string = node.to_string();
            if let Ok(serde_json::Value::Object(it)) = serde_json::from_str(&node_string) {
                let import_scope = ImportScope::find_insert_use_container(node, sema)?;
                let range = node.text_range();
                let mut edit = TextEdit::builder();
                edit.delete(range);
                let mut state = State::default();
                let semantics_scope = sema.scope(node)?;
                let scope_resolve =
                    |it| semantics_scope.speculative_resolve(&make::path_from_text(it));
                let scope_has = |it| scope_resolve(it).is_some();
                let deserialize_resolved = scope_resolve("::serde::Deserialize");
                let serialize_resolved = scope_resolve("::serde::Serialize");
                state.has_deserialize = deserialize_resolved.is_some();
                state.has_serialize = serialize_resolved.is_some();
                state.build_struct("Root", &it);
                edit.insert(range.start(), state.result);
                let vfs_file_id = file_id.file_id(sema.db);
                acc.push(
                    Diagnostic::new(
                        DiagnosticCode::Ra("json-is-not-rust", Severity::WeakWarning),
                        "JSON syntax is not valid as a Rust item",
                        FileRange { file_id: vfs_file_id, range },
                    )
                    .stable()
                    .with_fixes(Some(vec![{
                        let mut scb = SourceChangeBuilder::new(vfs_file_id);
                        let scope = scb.make_import_scope_mut(import_scope);
                        let current_module = semantics_scope.module();

                        let cfg = ImportPathConfig {
                            prefer_no_std: config.prefer_no_std,
                            prefer_prelude: config.prefer_prelude,
                            prefer_absolute: config.prefer_absolute,
                            allow_unstable: true,
                        };

                        if !scope_has("Serialize")
                            && let Some(PathResolution::Def(it)) = serialize_resolved
                            && let Some(it) = current_module.find_use_path(
                                sema.db,
                                it,
                                config.insert_use.prefix_kind,
                                cfg,
                            )
                        {
                            insert_use(&scope, mod_path_to_ast(&it, edition), &config.insert_use);
                        }
                        if !scope_has("Deserialize")
                            && let Some(PathResolution::Def(it)) = deserialize_resolved
                            && let Some(it) = current_module.find_use_path(
                                sema.db,
                                it,
                                config.insert_use.prefix_kind,
                                cfg,
                            )
                        {
                            insert_use(&scope, mod_path_to_ast(&it, edition), &config.insert_use);
                        }
                        let mut sc = scb.finish();
                        sc.insert_source_edit(vfs_file_id, edit.finish());
                        fix("convert_json_to_struct", "Convert JSON to struct", sc, range)
                    }])),
                );
            }
        }
        Some(())
    })();
}

#[cfg(test)]
mod tests {
    use crate::{
        DiagnosticsConfig,
        tests::{check_diagnostics_with_config, check_fix, check_no_fix},
    };

    #[test]
    fn diagnostic_for_simple_case() {
        let mut config = DiagnosticsConfig::test_sample();
        config.disabled.insert("syntax-error".to_owned());
        check_diagnostics_with_config(
            config,
            r#"
            { "foo": "bar" }
         // ^^^^^^^^^^^^^^^^ 💡 weak: JSON syntax is not valid as a Rust item
"#,
        );
    }

    #[test]
    fn types_of_primitives() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            use serde::Serialize;

            fn some_garbage() {

            }

            {$0
                "foo": "bar",
                "bar": 2.3,
                "baz": null,
                "bay": 57,
                "box": true
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;

            fn some_garbage() {

            }

            #[derive(Serialize)]
            struct Root1{ bar: f64, bay: i64, baz: (), r#box: bool, foo: String }

            "#,
        );
    }

    #[test]
    fn nested_structs() {
        check_fix(
            r#"
            {$0
                "foo": "bar",
                "bar": {
                    "kind": "Object",
                    "value": {}
                }
            }
            "#,
            r#"
            struct Value1{  }
            struct Bar1{ kind: String, value: Value1 }
            struct Root1{ bar: Bar1, foo: String }

            "#,
        );
    }

    #[test]
    fn naming() {
        check_fix(
            r#"
            {$0
                "user": {
                    "address": {
                        "street": "Main St",
                        "house": 3
                    },
                    "email": "example@example.com"
                },
                "another_user": {
                    "user": {
                        "address": {
                            "street": "Main St",
                            "house": 3
                        },
                        "email": "example@example.com"
                    }
                }
            }
            "#,
            r#"
            struct Address1{ house: i64, street: String }
            struct User1{ address: Address1, email: String }
            struct AnotherUser1{ user: User1 }
            struct Address2{ house: i64, street: String }
            struct User2{ address: Address2, email: String }
            struct Root1{ another_user: AnotherUser1, user: User2 }

            "#,
        );
    }

    #[test]
    fn arrays() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            {
                "of_string": ["foo", "2", "x"], $0
                "of_object": [{
                    "x": 10,
                    "y": 20
                }, {
                    "x": 10,
                    "y": 20
                }],
                "nested": [[[2]]],
                "empty": []
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            pub trait Deserialize {
                fn deserialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;
            use serde::Deserialize;

            #[derive(Serialize, Deserialize)]
            struct OfObject1{ x: i64, y: i64 }
            #[derive(Serialize, Deserialize)]
            struct Root1{ empty: Vec<_>, nested: Vec<Vec<Vec<i64>>>, of_object: Vec<OfObject1>, of_string: Vec<String> }

            "#,
        );
    }

    #[test]
    fn no_emit_outside_of_item_position() {
        check_no_fix(
            r#"
            fn foo() {
                let json = {$0
                    "foo": "bar",
                    "bar": {
                        "kind": "Object",
                        "value": {}
                    }
                };
            }
            "#,
        );
    }
}
