//! Runtime support needed for testing the stdsimd crate.
//!
//! This basically just disassembles the current executable and then parses the
//! output once globally and then provides the `assert` function which makes
//! assertions about the disassembly of a function.

#![cfg_attr(
    feature = "cargo-clippy",
    allow(missing_docs_in_private_items, print_stdout)
)]

extern crate assert_instr_macro;
extern crate backtrace;
extern crate cc;
#[macro_use]
extern crate lazy_static;
extern crate rustc_demangle;
extern crate simd_test_macro;
extern crate wasm_bindgen;

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process::Command;
use std::str;

use wasm_bindgen::prelude::*;

pub use assert_instr_macro::*;
pub use simd_test_macro::*;

lazy_static! {
    static ref DISASSEMBLY: HashMap<String, Vec<Function>> =
        disassemble_myself();
}

struct Function {
    addr: Option<usize>,
    instrs: Vec<Instruction>,
}

struct Instruction {
    parts: Vec<String>,
}

fn disassemble_myself() -> HashMap<String, Vec<Function>> {
    if cfg!(target_arch = "wasm32") {
        return parse_wasm2wat();
    }

    let me = env::current_exe().expect("failed to get current exe");

    if cfg!(target_arch = "x86_64")
        && cfg!(target_os = "windows")
        && cfg!(target_env = "msvc")
    {
        let mut cmd = cc::windows_registry::find(
            "x86_64-pc-windows-msvc",
            "dumpbin.exe",
        ).expect("failed to find `dumpbin` tool");
        let output = cmd
            .arg("/DISASM")
            .arg(&me)
            .output()
            .expect("failed to execute dumpbin");
        println!(
            "{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success());
        parse_dumpbin(&String::from_utf8_lossy(&output.stdout))
    } else if cfg!(target_os = "windows") {
        panic!("disassembly unimplemented")
    } else if cfg!(target_os = "macos") {
        let output = Command::new("otool")
            .arg("-vt")
            .arg(&me)
            .output()
            .expect("failed to execute otool");
        println!(
            "{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success());

        parse_otool(str::from_utf8(&output.stdout).expect("stdout not utf8"))
    } else {
        let objdump =
            env::var("OBJDUMP").unwrap_or_else(|_| "objdump".to_string());
        let output = Command::new(objdump.clone())
            .arg("--disassemble")
            .arg(&me)
            .output()
            .expect(&format!(
                "failed to execute objdump. OBJDUMP={}",
                objdump
            ));
        println!(
            "{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success());

        parse_objdump(str::from_utf8(&output.stdout).expect("stdout not utf8"))
    }
}

fn parse_objdump(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();
    let expected_len =
        if cfg!(target_arch = "arm") || cfg!(target_arch = "aarch64") {
            8
        } else {
            2
        };

    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut ret = HashMap::new();
    while let Some(header) = lines.next() {
        // symbols should start with `$hex_addr <$name>:`
        if !header.ends_with(">:") {
            continue;
        }
        let start = header.find('<')
            .expect(&format!("\"<\" not found in symbol pattern of the form \"$hex_addr <$name>\": {}", header));
        let symbol = &header[start + 1..header.len() - 2];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if instruction.is_empty() {
                break;
            }
            // Each line of instructions should look like:
            //
            //      $rel_offset: ab cd ef 00    $instruction...
            let parts = instruction
                .split_whitespace()
                .skip(1)
                .skip_while(|s| {
                    s.len() == expected_len
                        && usize::from_str_radix(s, 16).is_ok()
                }).map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert_with(Vec::new)
            .push(Function {
                addr: None,
                instrs: instructions,
            });
    }

    ret
}

fn parse_otool(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();

    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut ret = HashMap::new();
    let mut cached_header = None;
    while let Some(header) = cached_header.take().or_else(|| lines.next()) {
        // symbols should start with `$symbol:`
        if !header.ends_with(':') {
            continue;
        }
        // strip the leading underscore and the trailing colon
        let symbol = &header[1..header.len() - 1];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if instruction.ends_with(':') {
                cached_header = Some(instruction);
                break;
            }
            // Each line of instructions should look like:
            //
            //      $addr    $instruction...
            let parts = instruction
                .split_whitespace()
                .skip(1)
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert_with(Vec::new)
            .push(Function {
                addr: None,
                instrs: instructions,
            });
    }

    ret
}

fn parse_dumpbin(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();

    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut ret = HashMap::new();
    let mut cached_header = None;
    while let Some(header) = cached_header.take().or_else(|| lines.next()) {
        // symbols should start with `$symbol:`
        if !header.ends_with(':') {
            continue;
        }
        // strip the trailing colon
        let symbol = &header[..header.len() - 1];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if !instruction.starts_with("  ") {
                cached_header = Some(instruction);
                break;
            }
            // Each line looks like:
            //
            // >  $addr: ab cd ef     $instr..
            // >         00 12          # this line os optional
            if instruction.starts_with("       ") {
                continue;
            }
            let parts = instruction
                .split_whitespace()
                .skip(1)
                .skip_while(|s| {
                    s.len() == 2 && usize::from_str_radix(s, 16).is_ok()
                }).map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert_with(Vec::new)
            .push(Function {
                addr: None,
                instrs: instructions,
            });
    }

    ret
}

#[wasm_bindgen(module = "child_process")]
extern "C" {
    #[wasm_bindgen(js_name = execSync)]
    fn exec_sync(cmd: &str) -> Buffer;
}

#[wasm_bindgen(module = "buffer")]
extern "C" {
    type Buffer;
    #[wasm_bindgen(method, js_name = toString)]
    fn to_string(this: &Buffer) -> String;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = require)]
    fn resolve(module: &str) -> String;
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn js_console_log(s: &str);
}

// println! doesn't work on wasm32 right now, so shadow the compiler's println!
// macro with our own shim that redirects to `console.log`.
#[cfg(target_arch = "wasm32")]
macro_rules! println {
    ($($args:tt)*) => (js_console_log(&format!($($args)*)))
}

fn parse_wasm2wat() -> HashMap<String, Vec<Function>> {
    // Our wasm module in the wasm-bindgen test harness is called
    // "wasm-bindgen-test_bg". When running in node this is actually a shim JS
    // file. Ask node where that JS file is, and then we use that with a wasm
    // extension to find the wasm file itself.
    let js_shim = resolve("wasm-bindgen-test_bg");
    let js_shim = Path::new(&js_shim).with_extension("wasm");

    // Execute `wasm2wat` synchronously, waiting for and capturing all of its
    // output.
    let output =
        exec_sync(&format!("wasm2wat {}", js_shim.display())).to_string();

    let mut ret: HashMap<String, Vec<Function>> = HashMap::new();
    let mut lines = output.lines().map(|s| s.trim());
    while let Some(line) = lines.next() {
        // If we found the table of function pointers, fill in the known
        // address for all our `Function` instances
        if line.starts_with("(elem") {
            for (i, name) in line.split_whitespace().skip(3).enumerate() {
                let name = name.trim_right_matches(")");
                for f in ret.get_mut(name).expect("ret.get_mut(name) failed") {
                    f.addr = Some(i + 1);
                }
            }
            continue;
        }

        // If this isn't a function, we don't care about it.
        if !line.starts_with("(func ") {
            continue;
        }

        let mut function = Function {
            instrs: Vec::new(),
            addr: None,
        };

        // Empty functions will end in `))` so there's nothing to do, otherwise
        // we'll have a bunch of following lines which are instructions.
        //
        // Lines that have an imbalanced `)` mark the end of a function.
        if !line.ends_with("))") {
            while let Some(line) = lines.next() {
                function.instrs.push(Instruction {
                    parts: line
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect(),
                });
                if !line.starts_with("(") && line.ends_with(")") {
                    break;
                }
            }
        }

        // The second element here split on whitespace should be the name of
        // the function, skipping the type/params/results
        ret.entry(line.split_whitespace().nth(1).unwrap().to_string())
            .or_insert(Vec::new())
            .push(function);
    }
    return ret;
}

fn normalize(symbol: &str) -> String {
    let symbol = rustc_demangle::demangle(symbol).to_string();
    match symbol.rfind("::h") {
        Some(i) => symbol[..i].to_string(),
        None => symbol.to_string(),
    }
}

/// Main entry point for this crate, called by the `#[assert_instr]` macro.
///
/// This asserts that the function at `fnptr` contains the instruction
/// `expected` provided.
pub fn assert(fnptr: usize, fnname: &str, expected: &str) {
    let mut fnname = fnname.to_string();
    let functions = get_functions(fnptr, &mut fnname);
    assert_eq!(functions.len(), 1);
    let function = &functions[0];

    let mut instrs = &function.instrs[..];
    while instrs.last().map_or(false, |s| s.parts == ["nop"]) {
        instrs = &instrs[..instrs.len() - 1];
    }

    // Look for `expected` as the first part of any instruction in this
    // function, returning if we do indeed find it.
    let mut found = false;
    for instr in instrs {
        // Gets the first instruction, e.g. tzcntl in tzcntl %rax,%rax
        if let Some(part) = instr.parts.get(0) {
            // Truncates the instruction with the length of the expected
            // instruction: tzcntl => tzcnt and compares that.
            if part.starts_with(expected) {
                found = true;
                break;
            }
        }
    }

    // Look for `call` instructions in the disassembly to detect whether
    // inlining failed: all intrinsics are `#[inline(always)]`, so
    // calling one intrinsic from another should not generate `call`
    // instructions.
    let mut inlining_failed = false;
    for (i, instr) in instrs.iter().enumerate() {
        let part = match instr.parts.get(0) {
            Some(part) => part,
            None => continue,
        };
        if !part.contains("call") {
            continue;
        }

        // On 32-bit x86 position independent code will call itself and be
        // immediately followed by a `pop` to learn about the current address.
        // Let's not take that into account when considering whether a function
        // failed inlining something.
        let followed_by_pop = function
            .instrs
            .get(i + 1)
            .and_then(|i| i.parts.get(0))
            .map_or(false, |s| s.contains("pop"));
        if followed_by_pop && cfg!(target_arch = "x86") {
            continue;
        }

        inlining_failed = true;
        break;
    }

    let instruction_limit = std::env::var("STDSIMD_ASSERT_INSTR_LIMIT")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|_| match expected {
            // cpuid returns a pretty big aggregate structure so exempt it from
            // the slightly more restrictive 22 instructions below
            "cpuid" => 30,

            // Apparently on Windows LLVM generates a bunch of saves/restores
            // of xmm registers around these intstructions which
            // blows the 20 limit below. As it seems dictates by
            // Windows's abi (I guess?) we probably can't do much
            // about it...
            "vzeroall" | "vzeroupper" if cfg!(windows) => 30,

            // Intrinsics using `cvtpi2ps` are typically "composites" and in
            // some cases exceed the limit.
            "cvtpi2ps" => 25,

            // Original limit was 20 instructions, but ARM DSP Intrinsics are
            // exactly 20 instructions long. So bump the limit to 22 instead of
            // adding here a long list of exceptions.
            _ => 22,
        });
    let probably_only_one_instruction = instrs.len() < instruction_limit;

    if found && probably_only_one_instruction && !inlining_failed {
        return;
    }

    // Help debug by printing out the found disassembly, and then panic as we
    // didn't find the instruction.
    println!("disassembly for {}: ", fnname,);
    for (i, instr) in instrs.iter().enumerate() {
        let mut s = format!("\t{:2}: ", i);
        for part in &instr.parts {
            s.push_str(part);
            s.push_str(" ");
        }
        println!("{}", s);
    }

    if !found {
        panic!(
            "failed to find instruction `{}` in the disassembly",
            expected
        );
    } else if !probably_only_one_instruction {
        panic!(
            "instruction found, but the disassembly contains too many \
             instructions: #instructions = {} >= {} (limit)",
            instrs.len(),
            instruction_limit
        );
    } else if inlining_failed {
        panic!(
            "instruction found, but the disassembly contains `call` \
             instructions, which hint that inlining failed"
        );
    }
}

fn get_functions(fnptr: usize, fnname: &mut String) -> &'static [Function] {
    // Translate this function pointer to a symbolic name that we'd have found
    // in the disassembly.
    let mut sym = None;
    backtrace::resolve(fnptr as *mut _, |name| {
        sym = name.name().and_then(|s| s.as_str()).map(normalize);
    });

    if let Some(sym) = &sym {
        if let Some(s) = DISASSEMBLY.get(sym) {
            *fnname = sym.to_string();
            return s;
        }
    }

    let exact_match = DISASSEMBLY
        .iter()
        .find(|(_, list)| list.iter().any(|f| f.addr == Some(fnptr)));
    if let Some((name, list)) = exact_match {
        *fnname = name.to_string();
        return list;
    }

    if let Some(sym) = sym {
        println!("assumed symbol name: `{}`", sym);
    }
    println!("maybe related functions");
    for f in DISASSEMBLY.keys().filter(|k| k.contains(&**fnname)) {
        println!("\t- {}", f);
    }
    panic!("failed to find disassembly of {:#x} ({})", fnptr, fnname);
}

pub fn assert_skip_test_ok(name: &str) {
    if env::var("STDSIMD_TEST_EVERYTHING").is_err() {
        return;
    }
    panic!("skipped test `{}` when it shouldn't be skipped", name);
}

// See comment in `assert-instr-macro` crate for why this exists
pub static mut _DONT_DEDUP: &'static str = "";
