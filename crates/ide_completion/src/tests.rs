mod item_list;

use expect_test::Expect;
use stdx::format_to;

use crate::{
    test_utils::{self, get_all_items, TEST_CONFIG},
    CompletionConfig, CompletionItem,
};

fn completion_list(code: &str) -> String {
    completion_list_with_config(TEST_CONFIG, code)
}

fn completion_list_with_config(config: CompletionConfig, code: &str) -> String {
    fn monospace_width(s: &str) -> usize {
        s.chars().count()
    }

    let kind_completions: Vec<CompletionItem> = get_all_items(config, code).into_iter().collect();
    let label_width = kind_completions
        .iter()
        .map(|it| monospace_width(it.label()))
        .max()
        .unwrap_or_default()
        .min(16);
    kind_completions
        .into_iter()
        .map(|it| {
            let tag = it.kind().unwrap().tag();
            let var_name = format!("{} {}", tag, it.label());
            let mut buf = var_name;
            if let Some(detail) = it.detail() {
                let width = label_width.saturating_sub(monospace_width(it.label()));
                format_to!(buf, "{:width$} {}", "", detail, width = width);
            }
            if it.deprecated() {
                format_to!(buf, " DEPRECATED");
            }
            format_to!(buf, "\n");
            buf
        })
        .collect()
}

fn check(ra_fixture: &str, expect: Expect) {
    let base = r#"#[rustc_builtin_macro]
pub macro Clone {}
enum Enum { Variant }
struct Struct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();
trait Trait {}
"#;
    let actual = completion_list(&format!("{}{}", base, ra_fixture));
    expect.assert_eq(&actual)
}

fn check_no_completion(ra_fixture: &str) {
    let (db, position) = test_utils::position(ra_fixture);

    assert!(
        crate::completions(&db, &TEST_CONFIG, position).is_none(),
        "Completions were generated, but weren't expected"
    );
}

#[test]
fn test_no_completions_required() {
    cov_mark::check!(no_completion_required);
    check_no_completion(r#"fn foo() { for i i$0 }"#);
}
