use rustfix::{get_suggestions_from_json, Replacement};
use std::collections::HashSet;
use std::error::Error;

pub fn run_rustfix(code: &str, json: &str) -> String {
    let suggestions = get_suggestions_from_json(&json, &HashSet::new())
        .expect("could not load suggestions");

    let mut fixed = code.to_string();

    for sug in suggestions.into_iter().rev() {
        for sol in sug.solutions {
            for r in sol.replacements {
                fixed = apply_suggestion(&mut fixed, &r)
                    .expect("could not apply suggestion");
            }
        }
    }

    fixed
}

fn apply_suggestion(
    file_content: &mut String,
    suggestion: &Replacement,
) -> Result<String, Box<Error>> {
    use std::cmp::max;

    let mut new_content = String::new();

    // Add the lines before the section we want to replace
    new_content.push_str(&file_content
        .lines()
        .take(max(suggestion.snippet.line_range.start.line - 1, 0) as usize)
        .collect::<Vec<_>>()
        .join("\n"));
    new_content.push_str("\n");

    // Parts of line before replacement
    new_content.push_str(&file_content
        .lines()
        .nth(suggestion.snippet.line_range.start.line - 1)
        .unwrap_or("")
        .chars()
        .take(suggestion.snippet.line_range.start.column - 1)
        .collect::<String>());

    // Insert new content! Finally!
    new_content.push_str(&suggestion.replacement);

    // Parts of line after replacement
    new_content.push_str(&file_content
        .lines()
        .nth(suggestion.snippet.line_range.end.line - 1)
        .unwrap_or("")
        .chars()
        .skip(suggestion.snippet.line_range.end.column - 1)
        .collect::<String>());

    // Add the lines after the section we want to replace
    new_content.push_str("\n");
    new_content.push_str(&file_content
        .lines()
        .skip(suggestion.snippet.line_range.end.line as usize)
        .collect::<Vec<_>>()
        .join("\n"));
    new_content.push_str("\n");

    Ok(new_content)
}
