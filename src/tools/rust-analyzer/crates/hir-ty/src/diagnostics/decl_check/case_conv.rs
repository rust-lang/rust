//! Functions for string case manipulation, such as detecting the identifier case,
//! and converting it into appropriate form.

// Code that was taken from rustc was taken at commit 89fdb30,
// from file /compiler/rustc_lint/src/nonstandard_style.rs

/// Converts an identifier to an UpperCamelCase form.
/// Returns `None` if the string is already in UpperCamelCase.
pub(crate) fn to_camel_case(ident: &str) -> Option<String> {
    if is_camel_case(ident) {
        return None;
    }

    // Taken from rustc.
    let ret = ident
        .trim_matches('_')
        .split('_')
        .filter(|component| !component.is_empty())
        .map(|component| {
            let mut camel_cased_component = String::with_capacity(component.len());

            let mut new_word = true;
            let mut prev_is_lower_case = true;

            for c in component.chars() {
                // Preserve the case if an uppercase letter follows a lowercase letter, so that
                // `camelCase` is converted to `CamelCase`.
                if prev_is_lower_case && c.is_uppercase() {
                    new_word = true;
                }

                if new_word {
                    camel_cased_component.extend(c.to_uppercase());
                } else {
                    camel_cased_component.extend(c.to_lowercase());
                }

                prev_is_lower_case = c.is_lowercase();
                new_word = false;
            }

            camel_cased_component
        })
        .fold((String::new(), None), |(acc, prev): (_, Option<String>), next| {
            // separate two components with an underscore if their boundary cannot
            // be distinguished using an uppercase/lowercase case distinction
            let join = prev
                .and_then(|prev| {
                    let f = next.chars().next()?;
                    let l = prev.chars().last()?;
                    Some(!char_has_case(l) && !char_has_case(f))
                })
                .unwrap_or(false);
            (acc + if join { "_" } else { "" } + &next, Some(next))
        })
        .0;
    Some(ret)
}

/// Converts an identifier to a lower_snake_case form.
/// Returns `None` if the string is already in lower_snake_case.
pub(crate) fn to_lower_snake_case(ident: &str) -> Option<String> {
    if is_lower_snake_case(ident) {
        return None;
    } else if is_upper_snake_case(ident) {
        return Some(ident.to_lowercase());
    }

    Some(stdx::to_lower_snake_case(ident))
}

/// Converts an identifier to an UPPER_SNAKE_CASE form.
/// Returns `None` if the string is already is UPPER_SNAKE_CASE.
pub(crate) fn to_upper_snake_case(ident: &str) -> Option<String> {
    if is_upper_snake_case(ident) {
        return None;
    } else if is_lower_snake_case(ident) {
        return Some(ident.to_uppercase());
    }

    Some(stdx::to_upper_snake_case(ident))
}

// Taken from rustc.
// Modified by replacing the use of unstable feature `array_windows`.
fn is_camel_case(name: &str) -> bool {
    let name = name.trim_matches('_');
    if name.is_empty() {
        return true;
    }

    let mut fst = None;
    // start with a non-lowercase letter rather than non-uppercase
    // ones (some scripts don't have a concept of upper/lowercase)
    name.chars().next().map_or(true, |c| !c.is_lowercase())
        && !name.contains("__")
        && !name.chars().any(|snd| {
            let ret = match fst {
                None => false,
                Some(fst) => char_has_case(fst) && snd == '_' || char_has_case(snd) && fst == '_',
            };
            fst = Some(snd);

            ret
        })
}

fn is_lower_snake_case(ident: &str) -> bool {
    is_snake_case(ident, char::is_uppercase)
}

fn is_upper_snake_case(ident: &str) -> bool {
    is_snake_case(ident, char::is_lowercase)
}

// Taken from rustc.
// Modified to allow checking for both upper and lower snake case.
fn is_snake_case<F: Fn(char) -> bool>(ident: &str, wrong_case: F) -> bool {
    if ident.is_empty() {
        return true;
    }
    let ident = ident.trim_matches('_');

    let mut allow_underscore = true;
    ident.chars().all(|c| {
        allow_underscore = match c {
            '_' if !allow_underscore => return false,
            '_' => false,
            // It would be more obvious to check for the correct case,
            // but some characters do not have a case.
            c if !wrong_case(c) => true,
            _ => return false,
        };
        true
    })
}

// Taken from rustc.
fn char_has_case(c: char) -> bool {
    c.is_lowercase() || c.is_uppercase()
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
        check(to_lower_snake_case, "abc", expect![[""]]);
        check(to_lower_snake_case, "foo__bar", expect![["foo_bar"]]);
        check(to_lower_snake_case, "Δ", expect!["δ"]);
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
        // Taken from rustc: /compiler/rustc_lint/src/nonstandard_style/tests.rs
        check(to_camel_case, "X86_64", expect![[""]]);
        check(to_camel_case, "x86__64", expect![["X86_64"]]);
        check(to_camel_case, "Abc_123", expect![["Abc123"]]);
        check(to_camel_case, "A1_b2_c3", expect![["A1B2C3"]]);
    }

    #[test]
    fn test_to_upper_snake_case() {
        check(to_upper_snake_case, "UPPER_SNAKE_CASE", expect![[""]]);
        check(to_upper_snake_case, "lower_snake_case", expect![["LOWER_SNAKE_CASE"]]);
        check(to_upper_snake_case, "Weird_Case", expect![["WEIRD_CASE"]]);
        check(to_upper_snake_case, "CamelCase", expect![["CAMEL_CASE"]]);
        check(to_upper_snake_case, "lowerCamelCase", expect![["LOWER_CAMEL_CASE"]]);
        check(to_upper_snake_case, "A", expect![[""]]);
        check(to_upper_snake_case, "ABC", expect![[""]]);
        check(to_upper_snake_case, "X86_64", expect![[""]]);
        check(to_upper_snake_case, "FOO_BAr", expect![["FOO_BAR"]]);
        check(to_upper_snake_case, "FOO__BAR", expect![["FOO_BAR"]]);
        check(to_upper_snake_case, "ß", expect!["SS"]);
    }
}
