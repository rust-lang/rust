//! This diagnostic provides an assist for creating a struct definition from a JSON
//! example.

use ide_db::{base_db::FileId, source_change::SourceChange};
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    ast::{self, make},
    SyntaxKind, SyntaxNode,
};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, Severity};

#[derive(Default)]
struct State {
    result: String,
    struct_counts: usize,
}

impl State {
    fn generate_new_name(&mut self) -> ast::Name {
        self.struct_counts += 1;
        make::name(&format!("Struct{}", self.struct_counts))
    }

    fn build_struct(&mut self, value: &serde_json::Map<String, serde_json::Value>) -> ast::Type {
        let name = self.generate_new_name();
        let ty = make::ty(&name.to_string());
        let strukt = make::struct_(
            None,
            name,
            None,
            make::record_field_list(value.iter().sorted_unstable_by_key(|x| x.0).map(
                |(name, value)| make::record_field(None, make::name(name), self.type_of(value)),
            ))
            .into(),
        );
        format_to!(self.result, "#[derive(Serialize, Deserialize)]\n{}\n", strukt);
        ty
    }

    fn type_of(&mut self, value: &serde_json::Value) -> ast::Type {
        match value {
            serde_json::Value::Null => make::ty_unit(),
            serde_json::Value::Bool(_) => make::ty("bool"),
            serde_json::Value::Number(x) => make::ty(if x.is_i64() { "i64" } else { "f64" }),
            serde_json::Value::String(_) => make::ty("String"),
            serde_json::Value::Array(x) => {
                let ty = match x.iter().next() {
                    Some(x) => self.type_of(x),
                    None => make::ty_placeholder(),
                };
                make::ty(&format!("Vec<{ty}>"))
            }
            serde_json::Value::Object(x) => self.build_struct(x),
        }
    }
}

pub(crate) fn json_in_items(acc: &mut Vec<Diagnostic>, file_id: FileId, node: &SyntaxNode) {
    if node.kind() == SyntaxKind::ERROR
        && node.first_token().map(|x| x.kind()) == Some(SyntaxKind::L_CURLY)
        && node.last_token().map(|x| x.kind()) == Some(SyntaxKind::R_CURLY)
    {
        let node_string = node.to_string();
        if let Ok(x) = serde_json::from_str(&node_string) {
            if let serde_json::Value::Object(x) = x {
                let range = node.text_range();
                let mut edit = TextEdit::builder();
                edit.delete(range);
                let mut state = State::default();
                state.build_struct(&x);
                edit.insert(range.start(), state.result);
                acc.push(
                    Diagnostic::new(
                        "json-is-not-rust",
                        "JSON syntax is not valid as a Rust item",
                        range,
                    )
                    .severity(Severity::WeakWarning)
                    .with_fixes(Some(vec![fix(
                        "convert_json_to_struct",
                        "Convert JSON to struct",
                        SourceChange::from_text_edit(file_id, edit.finish()),
                        range,
                    )])),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_diagnostics_with_config, check_fix, check_no_fix},
        DiagnosticsConfig,
    };

    #[test]
    fn diagnostic_for_simple_case() {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("syntax-error".to_string());
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
            {$0
                "foo": "bar",
                "bar": 2.3,
                "baz": null,
                "bay": 57,
                "box": true
            }
            "#,
            r#"
            #[derive(Serialize, Deserialize)]
            struct Struct1{ bar: f64, bay: i64, baz: (), r#box: bool, foo: String }

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
            #[derive(Serialize, Deserialize)]
            struct Struct3{  }
            #[derive(Serialize, Deserialize)]
            struct Struct2{ kind: String, value: Struct3 }
            #[derive(Serialize, Deserialize)]
            struct Struct1{ bar: Struct2, foo: String }

            "#,
        );
    }

    #[test]
    fn arrays() {
        check_fix(
            r#"
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
            "#,
            r#"
            #[derive(Serialize, Deserialize)]
            struct Struct2{ x: i64, y: i64 }
            #[derive(Serialize, Deserialize)]
            struct Struct1{ empty: Vec<_>, nested: Vec<Vec<Vec<i64>>>, of_object: Vec<Struct2>, of_string: Vec<String> }

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
