#![feature(proc_macro)]

extern crate assert_instr_macro;
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
        let output = Command::new("objdump")
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

pub fn assert(fnptr: usize, expected: &str) {
    let mut sym = None;
    backtrace::resolve(fnptr as *mut _, |name| {
        sym = name.name().and_then(|s| s.as_str()).map(normalize);
    });

    let sym = match sym {
        Some(s) => s,
        None => panic!("failed to get symbol of function pointer: {}", fnptr),
    };

    let functions = &DISASSEMBLY.get(&sym)
        .expect(&format!("failed to find disassembly of {}", sym));
    assert_eq!(functions.len(), 1);
    let function = &functions[0];
    for instr in function.instrs.iter() {
        if let Some(part) = instr.parts.get(0) {
            if part == expected {
                return
            }
        }
    }

    println!("disassembly for {}: ", sym);
    for (i, instr) in function.instrs.iter().enumerate() {
        print!("\t{:2}: ", i);
        for part in instr.parts.iter() {
            print!("{} ", part);
        }
        println!("");
    }
    panic!("failed to find instruction `{}` in the disassembly", expected);
}

