use std::path::Path;

use crate::Suggestion;

type DynamicSuggestion = fn(&Path) -> Vec<Suggestion>;

pub(crate) const DYNAMIC_SUGGESTIONS: &[DynamicSuggestion] = &[|path: &Path| -> Vec<Suggestion> {
    if path.starts_with("compiler/") || path.starts_with("library/") {
        let path = path.components().take(2).collect::<Vec<_>>();

        vec![Suggestion::with_single_path(
            "test",
            None,
            &format!(
                "{}/{}",
                path[0].as_os_str().to_str().unwrap(),
                path[1].as_os_str().to_str().unwrap()
            ),
        )]
    } else {
        Vec::new()
    }
}];
