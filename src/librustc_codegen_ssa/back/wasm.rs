use std::fs;
use std::path::Path;
use std::str;

use serialize::leb128;

// https://webassembly.github.io/spec/core/binary/modules.html#binary-importsec
const WASM_CUSTOM_SECTION_ID: u8 = 0;

/// Adds or augment the existing `producers` section to encode information about
/// the Rust compiler used to produce the wasm file.
pub fn add_producer_section(
    path: &Path,
    rust_version: &str,
    rustc_version: &str,
) {
    struct Field<'a> {
        name: &'a str,
        values: Vec<FieldValue<'a>>,
    }

    #[derive(Copy, Clone)]
    struct FieldValue<'a> {
        name: &'a str,
        version: &'a str,
    }

    let wasm = fs::read(path).expect("failed to read wasm output");
    let mut ret = WasmEncoder::new();
    ret.data.extend(&wasm[..8]);

    // skip the 8 byte wasm/version header
    let rustc_value = FieldValue {
        name: "rustc",
        version: rustc_version,
    };
    let rust_value = FieldValue {
        name: "Rust",
        version: rust_version,
    };
    let mut fields = Vec::new();
    let mut wrote_rustc = false;
    let mut wrote_rust = false;

    // Move all sections from the original wasm file to our output, skipping
    // everything except the producers section
    for (id, raw) in WasmSections(WasmDecoder::new(&wasm[8..])) {
        if id != WASM_CUSTOM_SECTION_ID {
            ret.byte(id);
            ret.bytes(raw);
            continue
        }
        let mut decoder = WasmDecoder::new(raw);
        if decoder.str() != "producers" {
            ret.byte(id);
            ret.bytes(raw);
            continue
        }

        // Read off the producers section into our fields outside the loop,
        // we'll re-encode the producers section when we're done (to handle an
        // entirely missing producers section as well).
        info!("rewriting existing producers section");

        for _ in 0..decoder.u32() {
            let name = decoder.str();
            let mut values = Vec::new();
            for _ in 0..decoder.u32() {
                let name = decoder.str();
                let version = decoder.str();
                values.push(FieldValue { name, version });
            }

            if name == "language" {
                values.push(rust_value);
                wrote_rust = true;
            } else if name == "processed-by" {
                values.push(rustc_value);
                wrote_rustc = true;
            }
            fields.push(Field { name, values });
        }
    }

    if !wrote_rust {
        fields.push(Field {
            name: "language",
            values: vec![rust_value],
        });
    }
    if !wrote_rustc {
        fields.push(Field {
            name: "processed-by",
            values: vec![rustc_value],
        });
    }

    // Append the producers section to the end of the wasm file.
    let mut section = WasmEncoder::new();
    section.str("producers");
    section.u32(fields.len() as u32);
    for field in fields {
        section.str(field.name);
        section.u32(field.values.len() as u32);
        for value in field.values {
            section.str(value.name);
            section.str(value.version);
        }
    }
    ret.byte(WASM_CUSTOM_SECTION_ID);
    ret.bytes(&section.data);

    fs::write(path, &ret.data).expect("failed to write wasm output");
}

struct WasmSections<'a>(WasmDecoder<'a>);

impl<'a> Iterator for WasmSections<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<(u8, &'a [u8])> {
        if self.0.data.is_empty() {
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
}

struct WasmEncoder {
    data: Vec<u8>,
}

impl WasmEncoder {
    fn new() -> WasmEncoder {
        WasmEncoder { data: Vec::new() }
    }

    fn u32(&mut self, val: u32) {
        leb128::write_u32_leb128(&mut self.data, val);
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
}
