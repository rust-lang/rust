//! Runtime support needed for testing the stdsimd crate.
//!
//! This basically just disassembles the current executable and then parses the
//! output once globally and then provides the `assert` function which makes
//! assertions about the disassembly of a function.

#![feature(proc_macro)]

extern crate assert_instr_macro;
extern crate simd_test_macro;
extern crate backtrace;
extern crate cc;
extern crate rustc_demangle;
#[macro_use]
extern crate lazy_static;

use std::collections::HashMap;
use std::env;
use std::process::Command;
use std::str;

pub use assert_instr_macro::*;
pub use simd_test_macro::*;

lazy_static! {
    static ref DISASSEMBLY: HashMap<String, Vec<Function>> = disassemble_myself();
}

struct Function {
    instrs: Vec<Instruction>,
}

struct Instruction {
    parts: Vec<String>,
}

fn disassemble_myself() -> HashMap<String, Vec<Function>> {
    let me = env::current_exe().expect("failed to get current exe");

    if cfg!(target_arch = "x86_64") &&
        cfg!(target_os = "windows") &&
        cfg!(target_env = "msvc") {
        let mut cmd = cc::windows_registry::find("x86_64-pc-windows-msvc", "dumpbin.exe")
            .expect("failed to find `dumpbin` tool");
        let output = cmd.arg("/DISASM").arg(&me).output()
            .expect("failed to execute dumpbin");
        println!("{}\n{}", output.status, String::from_utf8_lossy(&output.stderr));
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
        println!("{}\n{}", output.status, String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success());

        parse_otool(&str::from_utf8(&output.stdout).expect("stdout not utf8"))
    } else {
        let objdump = env::var("OBJDUMP").unwrap_or("objdump".to_string());
        let output = Command::new(objdump)
            .arg("--disassemble")
            .arg(&me)
            .output()
            .expect("failed to execute objdump");
        println!("{}\n{}", output.status, String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success());

        parse_objdump(&str::from_utf8(&output.stdout).expect("stdout not utf8"))
    }
}

fn parse_objdump(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();
    let expected_len = if cfg!(target_arch = "arm") {
        8
    } else if cfg!(target_arch = "aarch64") {
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
            continue
        }
        let start = header.find("<").unwrap();
        let symbol = &header[start + 1..header.len() - 2];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if instruction.is_empty() {
                break
            }
            // Each line of instructions should look like:
            //
            //      $rel_offset: ab cd ef 00    $instruction...
            let parts = instruction.split_whitespace()
                .skip(1)
                .skip_while(|s| {
                    s.len() == expected_len && usize::from_str_radix(s, 16).is_ok()
                })
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert(Vec::new())
            .push(Function { instrs: instructions });
    }

    return ret
}

fn parse_otool(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();

    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut ret = HashMap::new();
    let mut cached_header = None;
    loop {
        let header = match cached_header.take().or_else(|| lines.next()) {
            Some(header) => header,
            None => break,
        };
        // symbols should start with `$symbol:`
        if !header.ends_with(":") {
            continue
        }
        // strip the leading underscore and the trailing colon
        let symbol = &header[1..header.len() - 1];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if instruction.ends_with(":") {
                cached_header = Some(instruction);
                break
            }
            // Each line of instructions should look like:
            //
            //      $addr    $instruction...
            let parts = instruction.split_whitespace()
                .skip(1)
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert(Vec::new())
            .push(Function { instrs: instructions });
    }

    return ret
}

fn parse_dumpbin(output: &str) -> HashMap<String, Vec<Function>> {
    let mut lines = output.lines();

    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut ret = HashMap::new();
    let mut cached_header = None;
    loop {
        let header = match cached_header.take().or_else(|| lines.next()) {
            Some(header) => header,
            None => break,
        };
        // symbols should start with `$symbol:`
        if !header.ends_with(":") {
            continue
        }
        // strip the trailing colon
        let symbol = &header[..header.len() - 1];

        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if !instruction.starts_with("  ") {
                cached_header = Some(instruction);
                break
            }
            // Each line looks like:
            //
            // >  $addr: ab cd ef     $instr..
            // >         00 12          # this line os optional
            if instruction.starts_with("       ") {
                continue
            }
            let parts = instruction.split_whitespace()
                .skip(1)
                .skip_while(|s| {
                    s.len() == 2 && usize::from_str_radix(s, 16).is_ok()
                })
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            instructions.push(Instruction { parts });
        }

        ret.entry(normalize(symbol))
            .or_insert(Vec::new())
            .push(Function { instrs: instructions });
    }

    return ret
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
    // Translate this function pointer to a symbolic name that we'd have found
    // in the disassembly.
    let mut sym = None;
    backtrace::resolve(fnptr as *mut _, |name| {
        sym = name.name().and_then(|s| s.as_str()).map(normalize);
    });

    let functions = match sym.as_ref().and_then(|s| DISASSEMBLY.get(s)) {
        Some(s) => s,
        None => {
            if let Some(sym) = sym {
                println!("assumed symbol name: `{}`", sym);
            }
            println!("maybe related functions");
            for f in DISASSEMBLY.keys().filter(|k| k.contains(fnname)) {
                println!("\t- {}", f);
            }
            panic!("failed to find disassembly of {:#x} ({})", fnptr, fnname);
        }
    };

    assert_eq!(functions.len(), 1);
    let function = &functions[0];

    // Look for `expected` as the first part of any instruction in this
    // function, returning if we do indeed find it.
    let mut found = false;
    for instr in function.instrs.iter() {
        // Gets the first instruction, e.g. tzcntl in tzcntl %rax,%rax
        if let Some(part) = instr.parts.get(0) {
            // Truncates the instruction with the length of the expected
            // instruction: tzcntl => tzcnt and compares that.
            if part.starts_with(expected) {
                found = true;
                break
            }
        }
    }

    let probably_only_one_instruction = function.instrs.len() < 30;

    if found && probably_only_one_instruction {
        return
    }

    // Help debug by printing out the found disassembly, and then panic as we
    // didn't find the instruction.
    println!("disassembly for {}: ", sym.as_ref().unwrap());
    for (i, instr) in function.instrs.iter().enumerate() {
        print!("\t{:2}: ", i);
        for part in instr.parts.iter() {
            print!("{} ", part);
        }
        println!("");
    }

    if !found {
        panic!("failed to find instruction `{}` in the disassembly", expected);
    } else if !probably_only_one_instruction {
        panic!("too many instructions in the disassembly");
    }
}
