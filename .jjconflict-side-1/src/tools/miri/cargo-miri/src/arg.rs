//! Utilities for dealing with argument flags

use std::borrow::Cow;
use std::env;

/// Determines whether a `--flag` is present.
pub fn has_arg_flag(name: &str) -> bool {
    num_arg_flag(name) > 0
}

/// Determines how many times a `--flag` is present.
pub fn num_arg_flag(name: &str) -> usize {
    env::args().take_while(|val| val != "--").filter(|val| val == name).count()
}

/// Yields all values of command line flag `name` as `Ok(arg)`, and all other arguments except
/// the flag as `Err(arg)`. (The flag `name` itself is not yielded at all, only its values are.)
pub struct ArgSplitFlagValue<'a, I> {
    args: Option<I>,
    name: &'a str,
}

impl<'a, I: Iterator> ArgSplitFlagValue<'a, I> {
    fn new(args: I, name: &'a str) -> Self {
        Self { args: Some(args), name }
    }
}

impl<'s, I: Iterator<Item = Cow<'s, str>>> Iterator for ArgSplitFlagValue<'_, I> {
    // If the original iterator was all `Owned`, then we will only ever yield `Owned`
    // (so `into_owned()` is cheap).
    type Item = Result<Cow<'s, str>, Cow<'s, str>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(args) = self.args.as_mut() else {
            // We already canceled this iterator.
            return None;
        };
        let arg = args.next()?;
        if arg == "--" {
            // Stop searching at `--`.
            self.args = None;
            // But yield the `--` so that it does not get lost!
            return Some(Err(Cow::Borrowed("--")));
        }
        // These branches cannot be merged if we want to avoid the allocation in the `Borrowed` branch.
        match &arg {
            Cow::Borrowed(arg) =>
                if let Some(suffix) = arg.strip_prefix(self.name) {
                    // Strip leading `name`.
                    if suffix.is_empty() {
                        // This argument is exactly `name`; the next one is the value.
                        return args.next().map(Ok);
                    } else if let Some(suffix) = suffix.strip_prefix('=') {
                        // This argument is `name=value`; get the value.
                        return Some(Ok(Cow::Borrowed(suffix)));
                    }
                },
            Cow::Owned(arg) =>
                if let Some(suffix) = arg.strip_prefix(self.name) {
                    // Strip leading `name`.
                    if suffix.is_empty() {
                        // This argument is exactly `name`; the next one is the value.
                        return args.next().map(Ok);
                    } else if let Some(suffix) = suffix.strip_prefix('=') {
                        // This argument is `name=value`; get the value. We need to do an allocation
                        // here as a `String` cannot be subsliced (what would the lifetime be?).
                        return Some(Ok(Cow::Owned(suffix.to_owned())));
                    }
                },
        }
        Some(Err(arg))
    }
}

impl<'a, I: Iterator<Item = String> + 'a> ArgSplitFlagValue<'a, I> {
    pub fn from_string_iter(
        args: I,
        name: &'a str,
    ) -> impl Iterator<Item = Result<String, String>> + 'a {
        ArgSplitFlagValue::new(args.map(Cow::Owned), name).map(|x| {
            match x {
                Ok(s) => Ok(s.into_owned()),
                Err(s) => Err(s.into_owned()),
            }
        })
    }
}

impl<'x: 'a, 'a, I: Iterator<Item = &'x str> + 'a> ArgSplitFlagValue<'a, I> {
    pub fn from_str_iter(
        args: I,
        name: &'a str,
    ) -> impl Iterator<Item = Result<&'x str, &'x str>> + 'a {
        ArgSplitFlagValue::new(args.map(Cow::Borrowed), name).map(|x| {
            match x {
                Ok(Cow::Borrowed(s)) => Ok(s),
                Err(Cow::Borrowed(s)) => Err(s),
                _ => panic!("iterator converted borrowed to owned"),
            }
        })
    }
}

/// Yields all values of command line flag `name`.
pub struct ArgFlagValueIter;

impl ArgFlagValueIter {
    pub fn from_string_iter<'a, I: Iterator<Item = String> + 'a>(
        args: I,
        name: &'a str,
    ) -> impl Iterator<Item = String> + 'a {
        ArgSplitFlagValue::from_string_iter(args, name).filter_map(Result::ok)
    }
}

impl ArgFlagValueIter {
    pub fn from_str_iter<'x: 'a, 'a, I: Iterator<Item = &'x str> + 'a>(
        args: I,
        name: &'a str,
    ) -> impl Iterator<Item = &'x str> + 'a {
        ArgSplitFlagValue::from_str_iter(args, name).filter_map(Result::ok)
    }
}

/// Gets the values of a `--flag`.
pub fn get_arg_flag_values(name: &str) -> impl Iterator<Item = String> + '_ {
    ArgFlagValueIter::from_string_iter(env::args(), name)
}

/// Gets the value of a `--flag`.
pub fn get_arg_flag_value(name: &str) -> Option<String> {
    get_arg_flag_values(name).next()
}
