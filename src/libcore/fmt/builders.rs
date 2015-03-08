// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use fmt::{self, Write, FlagV1};

struct PadAdapter<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    on_newline: bool,
}

impl<'a, 'b: 'a> PadAdapter<'a, 'b> {
    fn new(fmt: &'a mut fmt::Formatter<'b>) -> PadAdapter<'a, 'b> {
        PadAdapter {
            fmt: fmt,
            on_newline: false,
        }
    }
}

impl<'a, 'b: 'a> fmt::Write for PadAdapter<'a, 'b> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        while !s.is_empty() {
            if self.on_newline {
                try!(self.fmt.write_str("    "));
            }

            let split = match s.find('\n') {
                Some(pos) => {
                    self.on_newline = true;
                    pos + 1
                }
                None => {
                    self.on_newline = false;
                    s.len()
                }
            };
            try!(self.fmt.write_str(&s[..split]));
            s = &s[split..];
        }

        Ok(())
    }
}

/// A struct to help with `fmt::Debug` implementations.
///
/// Constructed by the `Formatter::debug_struct` method.
#[must_use]
pub struct DebugStruct<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    result: fmt::Result,
    has_fields: bool,
}

pub fn debug_struct_new<'a, 'b>(fmt: &'a mut fmt::Formatter<'b>, name: &str)
                                -> DebugStruct<'a, 'b> {
    let result = fmt.write_str(name);
    DebugStruct {
        fmt: fmt,
        result: result,
        has_fields: false,
    }
}

impl<'a, 'b: 'a> DebugStruct<'a, 'b> {
    /// Adds a new field to the generated struct output.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn field(mut self, name: &str, value: &fmt::Debug) -> DebugStruct<'a, 'b> {
        self.field_inner(name, value);
        self
    }

    #[inline(never)]
    fn field_inner(&mut self, name: &str, value: &fmt::Debug) {
        self.result = self.result.and_then(|_| {
            let prefix = if self.has_fields {
                ","
            } else {
                " {"
            };

            if self.is_pretty() {
                let mut writer = PadAdapter::new(self.fmt);
                fmt::write(&mut writer, format_args!("{}\n{}: {:#?}", prefix, name, value))
            } else {
                write!(self.fmt, "{} {}: {:?}", prefix, name, value)
            }
        });

        self.has_fields = true;
    }

    /// Consumes the `DebugStruct`, finishing output and returning any error
    /// encountered.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn finish(mut self) -> fmt::Result {
        self.finish_inner();
        self.result
    }

    #[inline(never)]
    fn finish_inner(&mut self) {
        if self.has_fields {
            self.result = self.result.and_then(|_| {
                if self.is_pretty() {
                    self.fmt.write_str("\n}")
                } else {
                    self.fmt.write_str(" }")
                }
            });
        }
    }

    fn is_pretty(&self) -> bool {
        self.fmt.flags() & (1 << (FlagV1::Alternate as usize)) != 0
    }
}

/// A struct to help with `fmt::Debug` implementations.
///
/// Constructed by the `Formatter::debug_tuple` method.
#[must_use]
pub struct DebugTuple<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    result: fmt::Result,
    has_fields: bool,
}

pub fn debug_tuple_new<'a, 'b>(fmt: &'a mut fmt::Formatter<'b>, name: &str) -> DebugTuple<'a, 'b> {
    let result = fmt.write_str(name);
    DebugTuple {
        fmt: fmt,
        result: result,
        has_fields: false,
    }
}

impl<'a, 'b: 'a> DebugTuple<'a, 'b> {
    /// Adds a new field to the generated tuple struct output.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn field(mut self, value: &fmt::Debug) -> DebugTuple<'a, 'b> {
        self.field_inner(value);
        self
    }

    #[inline(never)]
    fn field_inner(&mut self, value: &fmt::Debug) {
        self.result = self.result.and_then(|_| {
            let (prefix, space) = if self.has_fields {
                (",", " ")
            } else {
                ("(", "")
            };

            if self.is_pretty() {
                let mut writer = PadAdapter::new(self.fmt);
                fmt::write(&mut writer, format_args!("{}\n{:#?}", prefix, value))
            } else {
                write!(self.fmt, "{}{}{:?}", prefix, space, value)
            }
        });

        self.has_fields = true;
    }

    /// Consumes the `DebugTuple`, finishing output and returning any error
    /// encountered.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn finish(mut self) -> fmt::Result {
        self.finish_inner();
        self.result
    }

    #[inline(never)]
    fn finish_inner(&mut self) {
        if self.has_fields {
            self.result = self.result.and_then(|_| {
                if self.is_pretty() {
                    self.fmt.write_str("\n)")
                } else {
                    self.fmt.write_str(")")
                }
            });
        }
    }

    fn is_pretty(&self) -> bool {
        self.fmt.flags() & (1 << (FlagV1::Alternate as usize)) != 0
    }
}

/// A struct to help with `fmt::Debug` implementations.
///
/// Constructed by the `Formatter::debug_set` method.
#[must_use]
pub struct DebugSet<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    result: fmt::Result,
    has_fields: bool,
}

pub fn debug_set_new<'a, 'b>(fmt: &'a mut fmt::Formatter<'b>, name: &str) -> DebugSet<'a, 'b> {
    let result = write!(fmt, "{} {{", name);
    DebugSet {
        fmt: fmt,
        result: result,
        has_fields: false,
    }
}

impl<'a, 'b: 'a> DebugSet<'a, 'b> {
    /// Adds a new entry to the set output.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn entry(mut self, entry: &fmt::Debug) -> DebugSet<'a, 'b> {
        self.entry_inner(entry);
        self
    }

    #[inline(never)]
    fn entry_inner(&mut self, entry: &fmt::Debug) {
        self.result = self.result.and_then(|_| {
            let prefix = if self.has_fields {
                ","
            } else {
                ""
            };

            if self.is_pretty() {
                let mut writer = PadAdapter::new(self.fmt);
                fmt::write(&mut writer, format_args!("{}\n{:#?}", prefix, entry))
            } else {
                write!(self.fmt, "{} {:?}", prefix, entry)
            }
        });

        self.has_fields = true;
    }

    /// Consumes the `DebugSet`, finishing output and returning any error
    /// encountered.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn finish(mut self) -> fmt::Result {
        self.finish_inner();
        self.result
    }

    #[inline(never)]
    fn finish_inner(&mut self) {
        self.result = self.result.and_then(|_| {
            let end = match (self.has_fields, self.is_pretty()) {
                (false, _) => "}",
                (true, false) => " }",
                (true, true) => "\n}",
            };
            self.fmt.write_str(end)
        });
    }

    fn is_pretty(&self) -> bool {
        self.fmt.flags() & (1 << (FlagV1::Alternate as usize)) != 0
    }
}

/// A struct to help with `fmt::Debug` implementations.
///
/// Constructed by the `Formatter::debug_map` method.
#[must_use]
pub struct DebugMap<'a, 'b: 'a> {
    fmt: &'a mut fmt::Formatter<'b>,
    result: fmt::Result,
    has_fields: bool,
}

pub fn debug_map_new<'a, 'b>(fmt: &'a mut fmt::Formatter<'b>, name: &str) -> DebugMap<'a, 'b> {
    let result = write!(fmt, "{} {{", name);
    DebugMap {
        fmt: fmt,
        result: result,
        has_fields: false,
    }
}

impl<'a, 'b: 'a> DebugMap<'a, 'b> {
    /// Adds a new entry to the map output.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn entry(mut self, key: &fmt::Debug, value: &fmt::Debug) -> DebugMap<'a, 'b> {
        self.entry_inner(key, value);
        self
    }

    #[inline(never)]
    fn entry_inner(&mut self, key: &fmt::Debug, value: &fmt::Debug) {
        self.result = self.result.and_then(|_| {
            let prefix = if self.has_fields {
                ","
            } else {
                ""
            };

            if self.is_pretty() {
                let mut writer = PadAdapter::new(self.fmt);
                fmt::write(&mut writer, format_args!("{}\n{:#?}: {:#?}", prefix, key, value))
            } else {
                write!(self.fmt, "{} {:?}: {:?}", prefix, key, value)
            }
        });

        self.has_fields = true;
    }

    /// Consumes the `DebugMap`, finishing output and returning any error
    /// encountered.
    #[unstable(feature = "core", reason = "method was just created")]
    #[inline]
    pub fn finish(mut self) -> fmt::Result {
        self.finish_inner();
        self.result
    }

    #[inline(never)]
    fn finish_inner(&mut self) {
        self.result = self.result.and_then(|_| {
            let end = match (self.has_fields, self.is_pretty()) {
                (false, _) => "}",
                (true, false) => " }",
                (true, true) => "\n}",
            };
            self.fmt.write_str(end)
        });
    }

    fn is_pretty(&self) -> bool {
        self.fmt.flags() & (1 << (FlagV1::Alternate as usize)) != 0
    }
}
