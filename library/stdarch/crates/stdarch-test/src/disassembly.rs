//! Disassembly calling function for most targets.

use crate::Function;
use std::{collections::HashSet, env, str};

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
        symbol = symbol[last_colon + 1..].to_string();
    }

    // Normalize to no leading underscore to handle platforms that may
    // inject extra ones in symbol names.
    while symbol.starts_with('_') || symbol.starts_with('.') {
        symbol.remove(0);
    }
    // Windows/x86 has a suffix such as @@4.
    if let Some(idx) = symbol.find("@@") {
        symbol = symbol[..idx].to_string();
    }
    symbol
}

#[cfg(target_env = "msvc")]
pub(crate) fn disassemble_myself() -> HashSet<Function> {
    let me = env::current_exe().expect("failed to get current exe");

    let target = if cfg!(target_arch = "x86_64") {
        "x86_64-pc-windows-msvc"
    } else if cfg!(target_arch = "x86") {
        "i686-pc-windows-msvc"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64-pc-windows-msvc"
    } else {
        panic!("disassembly unimplemented")
    };
    let mut cmd =
        cc::windows_registry::find(target, "dumpbin.exe").expect("failed to find `dumpbin` tool");
    let output = cmd
        .arg("/DISASM:NOBYTES")
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
    parse(&String::from_utf8_lossy(Vec::leak(output.stdout)))
}

#[cfg(not(target_env = "msvc"))]
pub(crate) fn disassemble_myself() -> HashSet<Function> {
    let me = env::current_exe().expect("failed to get current exe");

    let objdump = env::var("OBJDUMP").unwrap_or_else(|_| "objdump".to_string());
    let add_args = if cfg!(target_vendor = "apple") && cfg!(target_arch = "aarch64") {
        // Target features need to be enabled for LLVM objdump on Darwin ARM64
        vec!["--mattr=+v8.6a,+crypto,+tme"]
    } else if cfg!(any(target_arch = "riscv32", target_arch = "riscv64")) {
        vec!["--mattr=+zk,+zks,+zbc,+zbb"]
    } else {
        vec![]
    };
    let output = std::process::Command::new(objdump.clone())
        .arg("--disassemble")
        .arg("--no-show-raw-insn")
        .args(add_args)
        .arg(&me)
        .output()
        .unwrap_or_else(|_| panic!("failed to execute objdump. OBJDUMP={objdump}"));
    println!(
        "{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success());

    let disassembly = String::from_utf8_lossy(Vec::leak(output.stdout));

    parse(&disassembly)
}

fn parse(output: &str) -> HashSet<Function> {
    let mut lines = output.lines();

    println!(
        "First 100 lines of the disassembly input containing {} lines:",
        lines.clone().count()
    );
    for line in output.lines().take(100) {
        println!("{line}");
    }

    let mut functions = HashSet::new();
    let mut cached_header = None;
    while let Some(header) = cached_header.take().or_else(|| lines.next()) {
        if !header.ends_with(':') || !header.contains("stdarch_test_shim") {
            continue;
        }
        eprintln!("header: {header}");
        let symbol = normalize(header);
        eprintln!("normalized symbol: {symbol}");
        let mut instructions = Vec::new();
        for instruction in lines.by_ref() {
            if instruction.ends_with(':') {
                cached_header = Some(instruction);
                break;
            }
            if instruction.is_empty() {
                cached_header = None;
                break;
            }
            let mut parts = if cfg!(target_env = "msvc") {
                // Each line looks like:
                //
                // >  $addr: $instr..
                instruction
                    .split(&[' ', ','])
                    .filter(|&x| !x.is_empty())
                    .skip(1)
                    .map(str::to_lowercase)
                    .skip_while(|s| matches!(&**s, "lock" | "vex")) // skip x86-specific prefix
                    .collect::<Vec<String>>()
            } else {
                // objdump with --no-show-raw-insn
                // Each line of instructions should look like:
                //
                //      $rel_offset:       $instruction...
                instruction
                    .split_whitespace()
                    .skip(1)
                    .skip_while(|s| matches!(*s, "lock" | "{evex}" | "{vex}")) // skip x86-specific prefix
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            };

            if cfg!(any(target_arch = "aarch64", target_arch = "arm64ec")) {
                // Normalize [us]shll.* ..., #0 instructions to the preferred form: [us]xtl.* ...
                // as neither LLVM objdump nor dumpbin does that.
                // See https://developer.arm.com/documentation/ddi0602/latest/SIMD-FP-Instructions/UXTL--UXTL2--Unsigned-extend-Long--an-alias-of-USHLL--USHLL2-
                // and https://developer.arm.com/documentation/ddi0602/latest/SIMD-FP-Instructions/SXTL--SXTL2--Signed-extend-Long--an-alias-of-SSHLL--SSHLL2-
                // for details.
                fn is_shll(instr: &str) -> bool {
                    if cfg!(target_env = "msvc") {
                        instr.starts_with("ushll") || instr.starts_with("sshll")
                    } else {
                        instr.starts_with("ushll.") || instr.starts_with("sshll.")
                    }
                }
                match (parts.first(), parts.last()) {
                    (Some(instr), Some(last_arg)) if is_shll(instr) && last_arg == "#0" => {
                        assert_eq!(parts.len(), 4);
                        let mut new_parts = Vec::with_capacity(3);
                        let new_instr = format!("{}{}{}", &instr[..1], "xtl", &instr[5..]);
                        new_parts.push(new_instr);
                        new_parts.push(parts[1].clone());
                        new_parts.push(parts[2][0..parts[2].len() - 1].to_owned()); // strip trailing comma
                        parts = new_parts;
                    }
                    // dumpbin uses "ins" instead of "mov"
                    (Some(instr), _) if cfg!(target_env = "msvc") && instr == "ins" => {
                        parts[0] = "mov".to_string()
                    }
                    _ => {}
                };
            }

            instructions.push(parts.join(" "));
            if matches!(&**instructions.last().unwrap(), "ret" | "retq") {
                cached_header = None;
                break;
            }
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
