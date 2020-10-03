pub fn to_camel_case(ident: &str) -> Option<String> {
    let mut output = String::new();

    if is_camel_case(ident) {
        return None;
    }

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
        None
    } else {
        Some(output)
    }
}

pub fn to_lower_snake_case(ident: &str) -> Option<String> {
    // First, assume that it's UPPER_SNAKE_CASE.
    if let Some(normalized) = to_lower_snake_case_from_upper_snake_case(ident) {
        return Some(normalized);
    }

    // Otherwise, assume that it's CamelCase.
    let lower_snake_case = stdx::to_lower_snake_case(ident);

    if lower_snake_case == ident {
        None
    } else {
        Some(lower_snake_case)
    }
}

fn to_lower_snake_case_from_upper_snake_case(ident: &str) -> Option<String> {
    if is_upper_snake_case(ident) {
        let string = ident.chars().map(|c| c.to_ascii_lowercase()).collect();
        Some(string)
    } else {
        None
    }
}

fn is_upper_snake_case(ident: &str) -> bool {
    ident.chars().all(|c| c.is_ascii_uppercase() || c == '_')
}

fn is_camel_case(ident: &str) -> bool {
    // We assume that the string is either snake case or camel case.
    ident.chars().all(|c| c != '_')
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
        check(to_lower_snake_case, "CamelCase", expect![["camel_case"]]);
    }

    #[test]
    fn test_to_camel_case() {
        check(to_camel_case, "CamelCase", expect![[""]]);
        check(to_camel_case, "lower_snake_case", expect![["LowerSnakeCase"]]);
        check(to_camel_case, "UPPER_SNAKE_CASE", expect![["UpperSnakeCase"]]);
    }
}
