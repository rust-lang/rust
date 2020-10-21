//! Functions for string case manipulation, such as detecting the identifier case,
//! and converting it into appropriate form.

#[derive(Debug)]
enum DetectedCase {
    LowerCamelCase,
    UpperCamelCase,
    LowerSnakeCase,
    UpperSnakeCase,
    Unknown,
}

fn detect_case(ident: &str) -> DetectedCase {
    let trimmed_ident = ident.trim_matches('_');
    let first_lowercase = trimmed_ident.starts_with(|chr: char| chr.is_ascii_lowercase());
    let mut has_lowercase = first_lowercase;
    let mut has_uppercase = false;
    let mut has_underscore = false;

    for chr in trimmed_ident.chars() {
        if chr == '_' {
            has_underscore = true;
        } else if chr.is_ascii_uppercase() {
            has_uppercase = true;
        } else if chr.is_ascii_lowercase() {
            has_lowercase = true;
        }
    }

    if has_uppercase {
        if !has_lowercase {
            if has_underscore {
                DetectedCase::UpperSnakeCase
            } else {
                // It has uppercase only and no underscores. Ex: "AABB"
                // This is a camel cased acronym.
                DetectedCase::UpperCamelCase
            }
        } else if !has_underscore {
            if first_lowercase {
                DetectedCase::LowerCamelCase
            } else {
                DetectedCase::UpperCamelCase
            }
        } else {
            // It has uppercase, it has lowercase, it has underscore.
            // No assumptions here
            DetectedCase::Unknown
        }
    } else {
        DetectedCase::LowerSnakeCase
    }
}

/// Converts an identifier to an UpperCamelCase form.
/// Returns `None` if the string is already is UpperCamelCase.
pub fn to_camel_case(ident: &str) -> Option<String> {
    let detected_case = detect_case(ident);

    match detected_case {
        DetectedCase::UpperCamelCase => return None,
        DetectedCase::LowerCamelCase => {
            let mut first_capitalized = false;
            let output = ident
                .chars()
                .map(|chr| {
                    if !first_capitalized && chr.is_ascii_lowercase() {
                        first_capitalized = true;
                        chr.to_ascii_uppercase()
                    } else {
                        chr
                    }
                })
                .collect();
            return Some(output);
        }
        _ => {}
    }

    let mut output = String::with_capacity(ident.len());

    let mut capital_added = false;
    for chr in ident.chars() {
        if chr.is_alphabetic() {
            if !capital_added {
                output.push(chr.to_ascii_uppercase());
                capital_added = true;
            } else {
                output.push(chr.to_ascii_lowercase());
            }
        } else if chr == '_' {
            // Skip this character and make the next one capital.
            capital_added = false;
        } else {
            // Put the characted as-is.
            output.push(chr);
        }
    }

    if output == ident {
        // While we didn't detect the correct case at the beginning, there
        // may be special cases: e.g. `A` is both valid CamelCase and UPPER_SNAKE_CASE.
        None
    } else {
        Some(output)
    }
}

/// Converts an identifier to a lower_snake_case form.
/// Returns `None` if the string is already in lower_snake_case.
pub fn to_lower_snake_case(ident: &str) -> Option<String> {
    // First, assume that it's UPPER_SNAKE_CASE.
    match detect_case(ident) {
        DetectedCase::LowerSnakeCase => return None,
        DetectedCase::UpperSnakeCase => {
            return Some(ident.chars().map(|chr| chr.to_ascii_lowercase()).collect())
        }
        _ => {}
    }

    // Otherwise, assume that it's CamelCase.
    let lower_snake_case = stdx::to_lower_snake_case(ident);

    if lower_snake_case == ident {
        // While we didn't detect the correct case at the beginning, there
        // may be special cases: e.g. `a` is both valid camelCase and snake_case.
        None
    } else {
        Some(lower_snake_case)
    }
}

/// Converts an identifier to an UPPER_SNAKE_CASE form.
/// Returns `None` if the string is already is UPPER_SNAKE_CASE.
pub fn to_upper_snake_case(ident: &str) -> Option<String> {
    match detect_case(ident) {
        DetectedCase::UpperSnakeCase => return None,
        DetectedCase::LowerSnakeCase => {
            return Some(ident.chars().map(|chr| chr.to_ascii_uppercase()).collect())
        }
        _ => {}
    }

    // Normalize the string from whatever form it's in currently, and then just make it uppercase.
    let upper_snake_case = stdx::to_upper_snake_case(ident);

    if upper_snake_case == ident {
        // While we didn't detect the correct case at the beginning, there
        // may be special cases: e.g. `A` is both valid CamelCase and UPPER_SNAKE_CASE.
        None
    } else {
        Some(upper_snake_case)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn check<F: Fn(&str) -> Option<String>>(fun: F, input: &str, expect: Expect) {
        // `None` is translated to empty string, meaning that there is nothing to fix.
        let output = fun(input).unwrap_or_default();

        expect.assert_eq(&output);
    }

    #[test]
    fn test_to_lower_snake_case() {
        check(to_lower_snake_case, "lower_snake_case", expect![[""]]);
        check(to_lower_snake_case, "UPPER_SNAKE_CASE", expect![["upper_snake_case"]]);
        check(to_lower_snake_case, "Weird_Case", expect![["weird_case"]]);
        check(to_lower_snake_case, "CamelCase", expect![["camel_case"]]);
        check(to_lower_snake_case, "lowerCamelCase", expect![["lower_camel_case"]]);
        check(to_lower_snake_case, "a", expect![[""]]);
    }

    #[test]
    fn test_to_camel_case() {
        check(to_camel_case, "CamelCase", expect![[""]]);
        check(to_camel_case, "CamelCase_", expect![[""]]);
        check(to_camel_case, "_CamelCase", expect![[""]]);
        check(to_camel_case, "lowerCamelCase", expect![["LowerCamelCase"]]);
        check(to_camel_case, "lower_snake_case", expect![["LowerSnakeCase"]]);
        check(to_camel_case, "UPPER_SNAKE_CASE", expect![["UpperSnakeCase"]]);
        check(to_camel_case, "Weird_Case", expect![["WeirdCase"]]);
        check(to_camel_case, "name", expect![["Name"]]);
        check(to_camel_case, "A", expect![[""]]);
        check(to_camel_case, "AABB", expect![[""]]);
    }

    #[test]
    fn test_to_upper_snake_case() {
        check(to_upper_snake_case, "UPPER_SNAKE_CASE", expect![[""]]);
        check(to_upper_snake_case, "lower_snake_case", expect![["LOWER_SNAKE_CASE"]]);
        check(to_upper_snake_case, "Weird_Case", expect![["WEIRD_CASE"]]);
        check(to_upper_snake_case, "CamelCase", expect![["CAMEL_CASE"]]);
        check(to_upper_snake_case, "lowerCamelCase", expect![["LOWER_CAMEL_CASE"]]);
        check(to_upper_snake_case, "A", expect![[""]]);
    }
}
