use std::{env, iter};

use anyhow::{Result, bail};

pub struct Args {
    args: iter::Peekable<env::Args>,
    /// Set to `true` once we saw a `--`.
    terminated: bool,
}

impl Args {
    pub fn new() -> Self {
        let mut args = Args { args: env::args().peekable(), terminated: false };
        args.args.next().unwrap(); // skip program name
        args
    }

    /// Get the next argument without any interpretation.
    pub fn next_raw(&mut self) -> Option<String> {
        self.args.next()
    }

    /// Consume a `-$f` flag if present.
    pub fn get_short_flag(&mut self, flag: char) -> Result<bool> {
        if self.terminated {
            return Ok(false);
        }
        if let Some(next) = self.args.peek() {
            if let Some(next) = next.strip_prefix("-") {
                if let Some(next) = next.strip_prefix(flag) {
                    if next.is_empty() {
                        self.args.next().unwrap(); // consume this argument
                        return Ok(true);
                    } else {
                        bail!("`-{flag}` followed by value");
                    }
                }
            }
        }
        Ok(false)
    }

    /// Consume a `--$name` flag if present.
    pub fn get_long_flag(&mut self, name: &str) -> Result<bool> {
        if self.terminated {
            return Ok(false);
        }
        if let Some(next) = self.args.peek() {
            if let Some(next) = next.strip_prefix("--") {
                if next == name {
                    self.args.next().unwrap(); // consume this argument
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Consume a `--$name val` or `--$name=val` option if present.
    pub fn get_long_opt(&mut self, name: &str) -> Result<Option<String>> {
        assert!(!name.is_empty());
        if self.terminated {
            return Ok(None);
        }
        let Some(next) = self.args.peek() else { return Ok(None) };
        let Some(next) = next.strip_prefix("--") else { return Ok(None) };
        let Some(next) = next.strip_prefix(name) else { return Ok(None) };
        // Starts with `--flag`.
        Ok(if let Some(val) = next.strip_prefix("=") {
            // `--flag=val` form
            let val = val.into();
            self.args.next().unwrap(); // consume this argument
            Some(val)
        } else if next.is_empty() {
            // `--flag val` form
            self.args.next().unwrap(); // consume this argument
            let Some(val) = self.args.next() else { bail!("`--{name}` not followed by value") };
            Some(val)
        } else {
            // Some unrelated flag, like `--flag-more` or so.
            None
        })
    }

    /// Consume a `--$name=val` or `--$name` option if present; the latter
    /// produces a default value. (`--$name val` is *not* accepted for this form
    /// of argument, it understands `val` already as the next argument!)
    pub fn get_long_opt_with_default(
        &mut self,
        name: &str,
        default: &str,
    ) -> Result<Option<String>> {
        assert!(!name.is_empty());
        if self.terminated {
            return Ok(None);
        }
        let Some(next) = self.args.peek() else { return Ok(None) };
        let Some(next) = next.strip_prefix("--") else { return Ok(None) };
        let Some(next) = next.strip_prefix(name) else { return Ok(None) };
        // Starts with `--flag`.
        Ok(if let Some(val) = next.strip_prefix("=") {
            // `--flag=val` form
            let val = val.into();
            self.args.next().unwrap(); // consume this argument
            Some(val)
        } else if next.is_empty() {
            // `--flag` form
            self.args.next().unwrap(); // consume this argument
            Some(default.into())
        } else {
            // Some unrelated flag, like `--flag-more` or so.
            None
        })
    }

    /// Returns the next free argument or uninterpreted flag, or `None` if there are no more
    /// arguments left. `--` is returned as well, but it is interpreted in the sense that no more
    /// flags will be parsed after this.
    pub fn get_other(&mut self) -> Option<String> {
        if self.terminated {
            return self.args.next();
        }
        let next = self.args.next()?;
        if next == "--" {
            self.terminated = true; // don't parse any more flags
            // This is where our parser is special, we do yield the `--`.
        }
        Some(next)
    }

    /// Return the rest of the aguments entirely unparsed.
    pub fn remainder(self) -> Vec<String> {
        self.args.collect()
    }
}
