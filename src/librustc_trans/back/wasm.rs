// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::str;

use rustc_data_structures::fx::FxHashMap;
use serialize::leb128;

// https://webassembly.github.io/spec/core/binary/modules.html#binary-importsec
const WASM_IMPORT_SECTION_ID: u8 = 2;

const WASM_EXTERNAL_KIND_FUNCTION: u8 = 0;
const WASM_EXTERNAL_KIND_TABLE: u8 = 1;
const WASM_EXTERNAL_KIND_MEMORY: u8 = 2;
const WASM_EXTERNAL_KIND_GLOBAL: u8 = 3;

/// Append all the custom sections listed in `sections` to the wasm binary
/// specified at `path`.
///
/// LLVM 6 which we're using right now doesn't have the ability to create custom
/// sections in wasm files nor does LLD have the ability to merge these sections
/// into one larger section when linking. It's expected that this will
/// eventually get implemented, however!
///
/// Until that time though this is a custom implementation in rustc to append
/// all sections to a wasm file to the finished product that LLD produces.
///
/// Support for this is landing in LLVM in https://reviews.llvm.org/D43097,
/// although after that support will need to be in LLD as well.
pub fn add_custom_sections(path: &Path, sections: &BTreeMap<String, Vec<u8>>) {
    if sections.len() == 0 {
        return
    }

    let wasm = fs::read(path).expect("failed to read wasm output");

    // see https://webassembly.github.io/spec/core/binary/modules.html#custom-section
    let mut wasm = WasmEncoder { data: wasm };
    for (section, bytes) in sections {
        // write the `id` identifier, 0 for a custom section
        wasm.byte(0);

        // figure out how long our name descriptor will be
        let mut name = WasmEncoder::new();
        name.str(section);

        // write the length of the payload followed by all its contents
        wasm.u32((bytes.len() + name.data.len()) as u32);
        wasm.data.extend_from_slice(&name.data);
        wasm.data.extend_from_slice(bytes);
    }

    fs::write(path, &wasm.data).expect("failed to write wasm output");
}

/// Rewrite the module imports are listed from in a wasm module given the field
/// name to module name mapping in `import_map`.
///
/// LLVM 6 which we're using right now doesn't have the ability to configure the
/// module a wasm symbol is import from. Rather all imported symbols come from
/// the bland `"env"` module unconditionally. Furthermore we'd *also* need
/// support in LLD for preserving these import modules, which it unfortunately
/// currently does not.
///
/// This function is intended as a hack for now where we manually rewrite the
/// wasm output by LLVM to have the correct import modules listed. The
/// `#[wasm_import_module]` attribute in Rust translates to the module that each
/// symbol is imported from, so here we manually go through the wasm file,
/// decode it, rewrite imports, and then rewrite the wasm module.
///
/// Support for this was added to LLVM in
/// https://github.com/llvm-mirror/llvm/commit/0f32e1365, although support still
/// needs to be added (AFAIK at the time of this writing) to LLD
pub fn rewrite_imports(path: &Path, import_map: &FxHashMap<String, String>) {
    if import_map.len() == 0 {
        return
    }

    let wasm = fs::read(path).expect("failed to read wasm output");
    let mut ret = WasmEncoder::new();
    ret.data.extend(&wasm[..8]);

    // skip the 8 byte wasm/version header
    for (id, raw) in WasmSections(WasmDecoder::new(&wasm[8..])) {
        ret.byte(id);
        if id == WASM_IMPORT_SECTION_ID {
            info!("rewriting import section");
            let data = rewrite_import_section(
                &mut WasmDecoder::new(raw),
                import_map,
            );
            ret.bytes(&data);
        } else {
            info!("carry forward section {}, {} bytes long", id, raw.len());
            ret.bytes(raw);
        }
    }

    fs::write(path, &ret.data).expect("failed to write wasm output");

    fn rewrite_import_section(
        wasm: &mut WasmDecoder,
        import_map: &FxHashMap<String, String>,
    )
        -> Vec<u8>
    {
        let mut dst = WasmEncoder::new();
        let n = wasm.u32();
        dst.u32(n);
        info!("rewriting {} imports", n);
        for _ in 0..n {
            rewrite_import_entry(wasm, &mut dst, import_map);
        }
        return dst.data
    }

    fn rewrite_import_entry(wasm: &mut WasmDecoder,
                            dst: &mut WasmEncoder,
                            import_map: &FxHashMap<String, String>) {
        // More info about the binary format here is available at:
        // https://webassembly.github.io/spec/core/binary/modules.html#import-section
        //
        // Note that you can also find the whole point of existence of this
        // function here, where we map the `module` name to a different one if
        // we've got one listed.
        let module = wasm.str();
        let field = wasm.str();
        let new_module = if module == "env" {
            import_map.get(field).map(|s| &**s).unwrap_or(module)
        } else {
            module
        };
        info!("import rewrite ({} => {}) / {}", module, new_module, field);
        dst.str(new_module);
        dst.str(field);
        let kind = wasm.byte();
        dst.byte(kind);
        match kind {
            WASM_EXTERNAL_KIND_FUNCTION => dst.u32(wasm.u32()),
            WASM_EXTERNAL_KIND_TABLE => {
                dst.byte(wasm.byte()); // element_type
                dst.limits(wasm.limits());
            }
            WASM_EXTERNAL_KIND_MEMORY => dst.limits(wasm.limits()),
            WASM_EXTERNAL_KIND_GLOBAL => {
                dst.byte(wasm.byte()); // content_type
                dst.bool(wasm.bool()); // mutable
            }
            b => panic!("unknown kind: {}", b),
        }
    }
}

struct WasmSections<'a>(WasmDecoder<'a>);

impl<'a> Iterator for WasmSections<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<(u8, &'a [u8])> {
        if self.0.data.len() == 0 {
            return None
        }

        // see https://webassembly.github.io/spec/core/binary/modules.html#sections
        let id = self.0.byte();
        let section_len = self.0.u32();
        info!("new section {} / {} bytes", id, section_len);
        let section = self.0.skip(section_len as usize);
        Some((id, section))
    }
}

struct WasmDecoder<'a> {
    data: &'a [u8],
}

impl<'a> WasmDecoder<'a> {
    fn new(data: &'a [u8]) -> WasmDecoder<'a> {
        WasmDecoder { data }
    }

    fn byte(&mut self) -> u8 {
        self.skip(1)[0]
    }

    fn u32(&mut self) -> u32 {
        let (n, l1) = leb128::read_u32_leb128(self.data);
        self.data = &self.data[l1..];
        return n
    }

    fn skip(&mut self, amt: usize) -> &'a [u8] {
        let (data, rest) = self.data.split_at(amt);
        self.data = rest;
        data
    }

    fn str(&mut self) -> &'a str {
        let len = self.u32();
        str::from_utf8(self.skip(len as usize)).unwrap()
    }

    fn bool(&mut self) -> bool {
        self.byte() == 1
    }

    fn limits(&mut self) -> (u32, Option<u32>) {
        let has_max = self.bool();
        (self.u32(), if has_max { Some(self.u32()) } else { None })
    }
}

struct WasmEncoder {
    data: Vec<u8>,
}

impl WasmEncoder {
    fn new() -> WasmEncoder {
        WasmEncoder { data: Vec::new() }
    }

    fn u32(&mut self, val: u32) {
        let at = self.data.len();
        leb128::write_u32_leb128(&mut self.data, at, val);
    }

    fn byte(&mut self, val: u8) {
        self.data.push(val);
    }

    fn bytes(&mut self, val: &[u8]) {
        self.u32(val.len() as u32);
        self.data.extend_from_slice(val);
    }

    fn str(&mut self, val: &str) {
        self.bytes(val.as_bytes())
    }

    fn bool(&mut self, b: bool) {
        self.byte(b as u8);
    }

    fn limits(&mut self, limits: (u32, Option<u32>)) {
        self.bool(limits.1.is_some());
        self.u32(limits.0);
        if let Some(c) = limits.1 {
            self.u32(c);
        }
    }
}
