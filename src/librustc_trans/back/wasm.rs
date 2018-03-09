// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs;
use std::path::Path;
use std::collections::BTreeMap;

use serialize::leb128;

pub fn add_custom_sections(path: &Path, sections: &BTreeMap<String, Vec<u8>>) {
    let mut wasm = fs::read(path).expect("failed to read wasm output");

    // see https://webassembly.github.io/spec/core/binary/modules.html#custom-section
    for (section, bytes) in sections {
        // write the `id` identifier, 0 for a custom section
        let len = wasm.len();
        leb128::write_u32_leb128(&mut wasm, len, 0);

        // figure out how long our name descriptor will be
        let mut name = Vec::new();
        leb128::write_u32_leb128(&mut name, 0, section.len() as u32);
        name.extend_from_slice(section.as_bytes());

        // write the length of the payload
        let len = wasm.len();
        let total_len = bytes.len() + name.len();
        leb128::write_u32_leb128(&mut wasm, len, total_len as u32);

        // write out the name section
        wasm.extend(name);

        // and now the payload itself
        wasm.extend_from_slice(bytes);
    }

    fs::write(path, &wasm).expect("failed to write wasm output");
}
