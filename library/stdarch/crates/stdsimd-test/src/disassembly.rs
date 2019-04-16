//! Disassembly calling function for most targets.

use ::*;
use std::process::Command;

pub(crate) fn disassemble_myself() -> HashMap<String, Vec<Function>> {
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
            .unwrap_or_else(|_| panic!(
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
            .unwrap_or_else(|| panic!("\"<\" not found in symbol pattern of the form \"$hex_addr <$name>\": {}", header));
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
                })
                .skip_while(|s| *s == "lock") // skip x86-specific prefix
                .map(std::string::ToString::to_string)
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
                .map(std::string::ToString::to_string)
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
                }).map(std::string::ToString::to_string)
                .skip_while(|s| *s == "lock") // skip x86-specific prefix
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
