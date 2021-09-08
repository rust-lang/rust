//! Disassembly calling function for most targets.

use crate::Function;
use std::{collections::HashSet, env, process::Command, str};

// Extracts the "shim" name from the `symbol`.
fn normalize(mut symbol: &str) -> String {
    // Remove trailing colon:
    if symbol.ends_with(':') {
        symbol = &symbol[..symbol.len() - 1];
    }
    if symbol.ends_with('>') {
        symbol = &symbol[..symbol.len() - 1];
    }
    if let Some(idx) = symbol.find('<') {
        symbol = &symbol[idx + 1..];
    }

    let mut symbol = rustc_demangle::demangle(symbol).to_string();
    symbol = match symbol.rfind("::h") {
        Some(i) => symbol[..i].to_string(),
        None => symbol.to_string(),
    };

    // Remove Rust paths
    if let Some(last_colon) = symbol.rfind(':') {
        symbol = (&symbol[last_colon + 1..]).to_string();
    }

    // Normalize to no leading underscore to handle platforms that may
    // inject extra ones in symbol names.
    while symbol.starts_with('_') {
        symbol.remove(0);
    }
    // Windows/x86 has a suffix such as @@4.
    if let Some(idx) = symbol.find("@@") {
        symbol = (&symbol[..idx]).to_string();
    }
    symbol
}

pub(crate) fn disassemble_myself() -> HashSet<Function> {
    let me = env::current_exe().expect("failed to get current exe");

    let disassembly = if cfg!(target_os = "windows") && cfg!(target_env = "msvc") {
        let target = if cfg!(target_arch = "x86_64") {
            "x86_64-pc-windows-msvc"
        } else if cfg!(target_arch = "x86") {
            "i686-pc-windows-msvc"
        } else {
            panic!("disassembly unimplemented")
        };
        let mut cmd = cc::windows_registry::find(target, "dumpbin.exe")
            .expect("failed to find `dumpbin` tool");
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
        // Windows does not return valid UTF-8 output:
        String::from_utf8_lossy(Vec::leak(output.stdout))
    } else if cfg!(target_os = "windows") {
        panic!("disassembly unimplemented")
    } else {
        let objdump = env::var("OBJDUMP").unwrap_or_else(|_| "objdump".to_string());
        let add_args = if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            // Target features need to be enabled for LLVM objdump on Macos ARM64
            vec!["--mattr=+v8.6a,+crypto,+tme"]
        } else {
            vec![]
        };
        let output = Command::new(objdump.clone())
            .arg("--disassemble")
            .arg("--no-show-raw-insn")
            .args(add_args)
            .arg(&me)
            .output()
            .unwrap_or_else(|_| panic!("failed to execute objdump. OBJDUMP={}", objdump));
        println!(
            "{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success());

        String::from_utf8_lossy(Vec::leak(output.stdout))
    };

    parse(&disassembly)
}

fn parse(output: &str) -> HashSet<Function> {
    let mut lines = output.lines();

    println!(
        "First 100 lines of the disassembly input containing {} lines:",
        lines.clone().count()
    );
    for line in output.lines().take(100) {
        println!("{}", line);
    }

    let mut functions = HashSet::new();
    let mut cached_header = None;
    while let Some(header) = cached_header.take().or_else(|| lines.next()) {
        if !header.ends_with(':') || !header.contains("stdarch_test_shim") {
            continue;
        }
        eprintln!("header: {}", header);
        let symbol = normalize(header);
        eprintln!("normalized symbol: {}", symbol);
        let mut instructions = Vec::new();
        while let Some(instruction) = lines.next() {
            if instruction.ends_with(':') {
                cached_header = Some(instruction);
                break;
            }
            if instruction.is_empty() {
                cached_header = None;
                break;
            }
            let parts = if cfg!(target_os = "macos") {
                // Each line of instructions should look like:
                //
                //      $addr    $instruction...
                instruction
                    .split_whitespace()
                    .skip(1)
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<String>>()
            } else if cfg!(target_env = "msvc") {
                // Each line looks like:
                //
                // >  $addr: ab cd ef     $instr..
                // >         00 12          # this line os optional
                if instruction.starts_with("       ") {
                    continue;
                }
                instruction
                    .split_whitespace()
                    .skip(1)
                    .skip_while(|s| s.len() == 2 && usize::from_str_radix(s, 16).is_ok())
                    .map(std::string::ToString::to_string)
                    .skip_while(|s| *s == "lock") // skip x86-specific prefix
                    .collect::<Vec<String>>()
            } else {
                // objdump with --no-show-raw-insn
                // Each line of instructions should look like:
                //
                //      $rel_offset:       $instruction...
                instruction
                    .split_whitespace()
                    .skip(1)
                    .skip_while(|s| *s == "lock") // skip x86-specific prefix
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<String>>()
            };
            instructions.push(parts.join(" "));
        }
        let function = Function {
            name: symbol,
            instrs: instructions,
        };
        assert!(functions.insert(function));
    }

    eprintln!("all found functions dump:");
    for k in &functions {
        eprintln!("  f: {}", k.name);
    }

    functions
}
