use std::fs;
use std::path::Path;
use std::str;

use rustc_data_structures::fx::FxHashMap;
use serialize::leb128;

// https://webassembly.github.io/spec/core/binary/modules.html#binary-importsec
const WASM_IMPORT_SECTION_ID: u8 = 2;
const WASM_CUSTOM_SECTION_ID: u8 = 0;

const WASM_EXTERNAL_KIND_FUNCTION: u8 = 0;
const WASM_EXTERNAL_KIND_TABLE: u8 = 1;
const WASM_EXTERNAL_KIND_MEMORY: u8 = 2;
const WASM_EXTERNAL_KIND_GLOBAL: u8 = 3;

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
/// `#[link(wasm_import_module = "...")]` attribute in Rust translates to the
/// module that each symbol is imported from, so here we manually go through the
/// wasm file, decode it, rewrite imports, and then rewrite the wasm module.
///
/// Support for this was added to LLVM in
/// https://github.com/llvm-mirror/llvm/commit/0f32e1365, although support still
/// needs to be added, tracked at https://bugs.llvm.org/show_bug.cgi?id=37168
pub fn rewrite_imports(path: &Path, import_map: &FxHashMap<String, String>) {
    if import_map.is_empty() {
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
