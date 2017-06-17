// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Write;

use rustc_serialize::json::as_json;

use Dump;

use rls_data::{Analysis, Import, Def, CratePreludeData, Format, Relation};


// A dumper to dump a restricted set of JSON information, designed for use with
// libraries distributed without their source. Clients are likely to use type
// information here, and (for example) generate Rustdoc URLs, but don't need
// information for navigating the source of the crate.
// Relative to the regular JSON save-analysis info, this form is filtered to
// remove non-visible items.
pub struct JsonApiDumper<'b, W: Write + 'b> {
    output: &'b mut W,
    result: Analysis,
}

impl<'b, W: Write> JsonApiDumper<'b, W> {
    pub fn new(writer: &'b mut W) -> JsonApiDumper<'b, W> {
        let mut result = Analysis::new();
        result.kind = Format::JsonApi;
        JsonApiDumper { output: writer, result }
    }
}

impl<'b, W: Write> Drop for JsonApiDumper<'b, W> {
    fn drop(&mut self) {
        if let Err(_) = write!(self.output, "{}", as_json(&self.result)) {
            error!("Error writing output");
        }
    }
}

impl<'b, W: Write + 'b> Dump for JsonApiDumper<'b, W> {
    fn crate_prelude(&mut self, data: CratePreludeData) {
        self.result.prelude = Some(data)
    }

    fn dump_relation(&mut self, data: Relation) {
        self.result.relations.push(data);
    }
    fn import(&mut self, public: bool, import: Import) {
        if public {
            self.result.imports.push(import);
        }
    }
    fn dump_def(&mut self, public: bool, mut data: Def) {
        if public {
            data.attributes = vec![];
            self.result.defs.push(data);
        }
    }
}
