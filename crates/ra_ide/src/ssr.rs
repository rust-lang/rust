use ra_db::FilePosition;
use ra_ide_db::RootDatabase;

use crate::SourceFileEdit;
use ra_ssr::{MatchFinder, SsrError, SsrRule};

// Feature: Structural Search and Replace
//
// Search and replace with named wildcards that will match any expression, type, path, pattern or item.
// The syntax for a structural search replace command is `<search_pattern> ==>> <replace_pattern>`.
// A `$<name>` placeholder in the search pattern will match any AST node and `$<name>` will reference it in the replacement.
// Within a macro call, a placeholder will match up until whatever token follows the placeholder.
//
// Placeholders may be given constraints by writing them as `${<name>:<constraint1>:<constraint2>...}`.
//
// Supported constraints:
//
// |===
// | Constraint    | Restricts placeholder
//
// | kind(literal) | Is a literal (e.g. `42` or `"forty two"`)
// | not(a)        | Negates the constraint `a`
// |===
//
// Available via the command `rust-analyzer.ssr`.
//
// ```rust
// // Using structural search replace command [foo($a, $b) ==>> ($a).foo($b)]
//
// // BEFORE
// String::from(foo(y + 5, z))
//
// // AFTER
// String::from((y + 5).foo(z))
// ```
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Structural Search Replace**
// |===
pub fn parse_search_replace(
    rule: &str,
    parse_only: bool,
    db: &RootDatabase,
    position: FilePosition,
) -> Result<Vec<SourceFileEdit>, SsrError> {
    let rule: SsrRule = rule.parse()?;
    let mut match_finder = MatchFinder::in_context(db, position);
    match_finder.add_rule(rule);
    if parse_only {
        return Ok(Vec::new());
    }
    Ok(match_finder.edits())
}
