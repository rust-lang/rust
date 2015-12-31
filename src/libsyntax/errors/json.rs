// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A JSON emitter for errors.

use codemap::{Span, CodeMap};
use diagnostics::registry::Registry;
use errors::{Level, DiagnosticBuilder, RenderSpan};
use errors::emitter::Emitter;

use std::rc::Rc;

pub struct JsonEmitter {
    todo: i32
}

impl JsonEmitter {
    pub fn basic() -> JsonEmitter {
        JsonEmitter {
            todo: 42,
        }
    }

    pub fn stderr(registry: Option<Registry>,
                  code_map: Rc<CodeMap>) -> JsonEmitter {
        JsonEmitter {
            todo: 42,
        }
    }
}

impl Emitter for JsonEmitter {
    fn emit(&mut self, span: Option<Span>, msg: &str, code: Option<&str>, lvl: Level) {
        unimplemented!();

    }

    fn custom_emit(&mut self, sp: RenderSpan, msg: &str, lvl: Level) {
        unimplemented!();

    }

    fn emit_struct(&mut self, db: &DiagnosticBuilder) {
        unimplemented!();
    }
}
