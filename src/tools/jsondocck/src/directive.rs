use std::borrow::Cow;
use std::fmt::Debug;

use jaq_json::Val;
use serde_json::Value;

use crate::cache::Cache;

#[derive(Debug)]
pub struct Directive {
    pub kind: DirectiveKind,
    pub path: String,
    pub lineno: usize,
}

#[derive(Debug)]
pub enum DirectiveKind {
    JsonPath(JsonPathDirective),
    Jq(JqDirective),
}

#[derive(Debug)]
pub enum JsonPathDirective {
    /// `//@ has <path>`
    ///
    /// Checks the path exists.
    HasPath,

    /// `//@ has <path> <value>`
    ///
    /// Check one thing at the path  is equal to the value.
    HasValue { value: String },

    /// `//@ !has <path>`
    ///
    /// Checks the path doesn't exist.
    HasNotPath,

    /// `//@ !has <path> <value>`
    ///
    /// Checks the path exists, but doesn't have the given value.
    HasNotValue { value: String },

    /// `//@ is <path> <value>`
    ///
    /// Check the path is the given value.
    Is { value: String },

    /// `//@ is <path> <value> <value>...`
    ///
    /// Check that the path matches to exactly every given value.
    IsMany { values: Vec<String> },

    /// `//@ !is <path> <value>`
    ///
    /// Check the path isn't the given value.
    IsNot { value: String },

    /// `//@ count <path> <value>`
    ///
    /// Check the path has the expected number of matches.
    CountIs { expected: usize },

    /// `//@ set <name> = <path>`
    Set { variable: String },
}

#[derive(Debug)]
pub enum JqDirective {
    /// `//@ jq_has <path> <value>`
    ///
    /// Check one thing at the path is equal to the value.
    HasValue { value: String },

    /// `//@ !jq_has <path> <value>`
    ///
    /// Checks the path exists, but doesn't have the given value.
    HasNotValue { value: String },

    /// `//@ jq_is <path> <value>`
    ///
    /// Check the path is the given value.
    Is { value: String },

    /// `//@ jq_is <path> <value> <value>...`
    ///
    /// Check that the path matches to exactly every given value.
    IsMany { values: Vec<String> },

    /// `//@ !jq_is <path> <value>`
    ///
    /// Check the path isn't the given value.
    IsNot { value: String },

    /// `//@ jq_count <path> <value>`
    ///
    /// Check the path has the expected number of matches.
    CountIs { expected: usize },

    /// `//@ jq_set <name> = <path>`
    Set { variable: String },
}

impl DirectiveKind {
    /// Returns both the kind and the path.
    ///
    /// Returns `None` if the directive isn't from jsondocck (e.g. from compiletest).
    pub fn parse<'a>(
        directive_name: &str,
        negated: bool,
        args: &'a [String],
    ) -> Option<(Self, &'a str)> {
        let kind = match (directive_name, negated) {
            // jsonpath
            ("count", false) => {
                assert_eq!(args.len(), 2);
                let expected = args[1].parse().expect("invalid number for `count`");
                Self::JsonPath(JsonPathDirective::CountIs { expected })
            }
            ("ismany", false) => {
                // FIXME: Make this >= 3, and migrate len(values)==1 cases to @is
                assert!(args.len() >= 2, "Not enough args to `ismany`");
                let values = args[1..].to_owned();
                Self::JsonPath(JsonPathDirective::IsMany { values })
            }
            ("is", false) => {
                assert_eq!(args.len(), 2);
                Self::JsonPath(JsonPathDirective::Is { value: args[1].clone() })
            }
            ("is", true) => {
                assert_eq!(args.len(), 2);
                Self::JsonPath(JsonPathDirective::IsNot { value: args[1].clone() })
            }
            ("set", false) => {
                assert_eq!(args.len(), 3);
                assert_eq!(args[1], "=");
                return Some((
                    Self::JsonPath(JsonPathDirective::Set { variable: args[0].clone() }),
                    &args[2],
                ));
            }
            ("has", false) => match args {
                [_path] => Self::JsonPath(JsonPathDirective::HasPath),
                [_path, value] => {
                    Self::JsonPath(JsonPathDirective::HasValue { value: value.clone() })
                }
                _ => panic!("`//@ has` must have 2 or 3 arguments, but got {args:?}"),
            },
            ("has", true) => match args {
                [_path] => Self::JsonPath(JsonPathDirective::HasNotPath),
                [_path, value] => {
                    Self::JsonPath(JsonPathDirective::HasNotValue { value: value.clone() })
                }
                _ => panic!("`//@ !has` must have 2 or 3 arguments, but got {args:?}"),
            },

            // jq
            ("jq_count", false) => {
                assert_eq!(args.len(), 2);
                let expected = args[1].parse().expect("invalid number for `count`");
                Self::Jq(JqDirective::CountIs { expected })
            }
            ("jq_ismany", false) => {
                // FIXME: Make this >= 3, and migrate len(values)==1 cases to @is
                assert!(args.len() >= 2, "Not enough args to `ismany`");
                let values = args[1..].to_owned();
                Self::Jq(JqDirective::IsMany { values })
            }
            ("jq_is", false) => {
                assert_eq!(args.len(), 2);
                Self::Jq(JqDirective::Is { value: args[1].clone() })
            }
            ("jq_is", true) => {
                assert_eq!(args.len(), 2);
                Self::Jq(JqDirective::IsNot { value: args[1].clone() })
            }
            ("jq_has", false) => {
                assert_eq!(args.len(), 2);
                Self::Jq(JqDirective::HasValue { value: args[1].clone() })
            }
            ("jq_has", true) => {
                assert_eq!(args.len(), 2);
                Self::Jq(JqDirective::HasNotValue { value: args[1].clone() })
            }
            ("jq_set", false) => {
                assert_eq!(args.len(), 3);
                assert_eq!(args[1], "=");
                return Some((Self::Jq(JqDirective::Set { variable: args[0].clone() }), &args[2]));
            }
            // Ignore unknown directives as they might be compiletest directives
            // since they share the same `//@` prefix by convention. In any case,
            // compiletest rejects unknown directives for us.
            _ => return None,
        };

        Some((kind, &args[0]))
    }
}

impl Directive {
    /// Performs the actual work of ensuring a directive passes.
    pub fn check(&self, cache: &mut Cache) -> Result<(), String> {
        match &self.kind {
            DirectiveKind::JsonPath(d) => {
                let matches = cache.select(&self.path);
                match d {
                    JsonPathDirective::HasPath => {
                        if matches.is_empty() {
                            return Err("matched to no values".to_owned());
                        }
                    }
                    JsonPathDirective::HasNotPath => {
                        if !matches.is_empty() {
                            return Err(format!("matched to {matches:?}, but wanted no matches"));
                        }
                    }
                    JsonPathDirective::HasValue { value } => {
                        let want_value = string_to_value(value, cache);
                        if !matches.contains(&want_value.as_ref()) {
                            return Err(format!(
                                "matched to {matches:?}, which didn't contain {want_value:?}"
                            ));
                        }
                    }
                    JsonPathDirective::HasNotValue { value } => {
                        let wantnt_value = string_to_value(value, cache);
                        if matches.contains(&wantnt_value.as_ref()) {
                            return Err(format!(
                                "matched to {matches:?}, which contains unwanted {wantnt_value:?}"
                            ));
                        } else if matches.is_empty() {
                            return Err(format!(
                                "got no matches, but expected some matched (not containing {wantnt_value:?}"
                            ));
                        }
                    }

                    JsonPathDirective::Is { value } => {
                        let want_value = string_to_value(value, cache);
                        let matched = get_one(&matches)?;
                        if matched != want_value.as_ref() {
                            return Err(format!("matched to {matched:?} but want {want_value:?}"));
                        }
                    }
                    JsonPathDirective::IsNot { value } => {
                        let wantnt_value = string_to_value(value, cache);
                        let matched = get_one(&matches)?;
                        if matched == wantnt_value.as_ref() {
                            return Err(format!(
                                "got value {wantnt_value:?}, but want anything else"
                            ));
                        }
                    }

                    JsonPathDirective::IsMany { values } => {
                        // Serde json doesn't implement Ord or Hash for Value, so we must
                        // use a Vec here. While in theory that makes setwize equality
                        // O(n^2), in practice n will never be large enough to matter.
                        let expected_values =
                            values.iter().map(|v| string_to_value(v, cache)).collect::<Vec<_>>();
                        if expected_values.len() != matches.len() {
                            return Err(format!(
                                "Expected {} values, but matched to {} values ({:?})",
                                expected_values.len(),
                                matches.len(),
                                matches
                            ));
                        };
                        for got_value in matches {
                            if !expected_values.iter().any(|exp| &**exp == got_value) {
                                return Err(format!(
                                    "has match {got_value:?}, which was not expected",
                                ));
                            }
                        }
                    }
                    JsonPathDirective::CountIs { expected } => {
                        if *expected != matches.len() {
                            return Err(format!(
                                "matched to `{matches:?}` with length {}, but expected length {expected}",
                                matches.len(),
                            ));
                        }
                    }
                    JsonPathDirective::Set { variable } => {
                        let value = get_one(&matches)?;
                        let r = cache.variables.insert(variable.to_owned(), value.clone());
                        assert!(
                            r.is_none() && !cache.jq_variables.contains_key(variable),
                            "name collision: {variable:?} is duplicated"
                        );
                    }
                }
            }
            DirectiveKind::Jq(d) => {
                let matches = cache.jq_select(&self.path);
                match d {
                    JqDirective::HasValue { value } => {
                        let want_value = string_to_val(value, cache);
                        if !matches.contains(&want_value.as_ref()) {
                            return Err(format!(
                                "matched to {matches:?}, which didn't contain {want_value:?}"
                            ));
                        }
                    }
                    JqDirective::HasNotValue { value } => {
                        let wantnt_value = string_to_val(value, cache);
                        if matches.contains(&wantnt_value.as_ref()) {
                            return Err(format!(
                                "matched to {matches:?}, which contains unwanted {wantnt_value:?}"
                            ));
                        } else if matches.is_empty() {
                            return Err(format!(
                                "got no matches, but expected some matched (not containing {wantnt_value:?}"
                            ));
                        }
                    }

                    JqDirective::Is { value } => {
                        let want_value = string_to_val(value, cache);
                        let matched = get_one(&matches.iter().collect::<Vec<&Val>>())?;
                        if matched != want_value.as_ref() {
                            return Err(format!("matched to {matched:?} but want {want_value:?}"));
                        }
                    }
                    JqDirective::IsNot { value } => {
                        let wantnt_value = string_to_val(value, cache);
                        let matched = get_one(&matches.iter().collect::<Vec<&Val>>())?;
                        if matched == wantnt_value.as_ref() {
                            return Err(format!(
                                "got value {wantnt_value:?}, but want anything else"
                            ));
                        }
                    }

                    JqDirective::IsMany { values } => {
                        // Serde json doesn't implement Ord or Hash for Value, so we must
                        // use a Vec here. While in theory that makes setwize equality
                        // O(n^2), in practice n will never be large enough to matter.
                        let expected_values =
                            values.iter().map(|v| string_to_val(v, cache)).collect::<Vec<_>>();
                        if expected_values.len() != matches.len() {
                            return Err(format!(
                                "Expected {} values, but matched to {} values ({:?})",
                                expected_values.len(),
                                matches.len(),
                                matches
                            ));
                        };
                        for got_value in matches {
                            if !expected_values.iter().any(|exp| **exp == got_value) {
                                return Err(format!(
                                    "has match {got_value:?}, which was not expected",
                                ));
                            }
                        }
                    }
                    JqDirective::CountIs { expected } => {
                        if *expected != matches.len() {
                            return Err(format!(
                                "matched to `{matches:?}` with length {}, but expected length {expected}",
                                matches.len(),
                            ));
                        }
                    }
                    JqDirective::Set { variable } => {
                        let value = get_one(&matches.iter().collect::<Vec<&Val>>())?;
                        let r = cache.jq_variables.insert(variable.to_owned(), value.clone());
                        assert!(
                            r.is_none() && !cache.variables.contains_key(variable),
                            "name collision: {variable:?} is duplicated"
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

fn get_one<'a, T: Debug>(matches: &[&'a T]) -> Result<&'a T, String> {
    match matches {
        [] => Err("matched to no values".to_owned()),
        [matched] => Ok(matched),
        _ => Err(format!("matched to multiple values {matches:?}, but want exactly 1")),
    }
}

fn string_to_value<'a>(s: &str, cache: &'a Cache) -> Cow<'a, Value> {
    if s.starts_with("$") {
        Cow::Borrowed(&cache.variables.get(&s[1..]).unwrap_or_else(|| {
            // FIXME(adotinthevoid): Show line number
            panic!("No variable: `{}`. Current state: `{:?}`", &s[1..], cache.variables)
        }))
    } else {
        Cow::Owned(serde_json::from_str(s).expect(&format!("Cannot convert `{}` to json", s)))
    }
}

fn string_to_val<'a>(s: &str, cache: &'a Cache) -> Cow<'a, Val> {
    if s.starts_with("$") {
        Cow::Borrowed(&cache.jq_variables.get(&s[1..]).unwrap_or_else(|| {
            // FIXME(adotinthevoid): Show line number
            panic!("No variable: `{}`. Current state: `{:?}`", &s[1..], cache.jq_variables)
        }))
    } else {
        Cow::Owned(serde_json::from_str(s).expect(&format!("Cannot convert `{}` to json", s)))
    }
}
