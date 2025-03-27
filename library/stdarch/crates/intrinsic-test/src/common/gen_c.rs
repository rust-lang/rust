use itertools::Itertools;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::process::Command;

pub fn generate_c_program(
    notices: &str,
    header_files: &[&str],
    arch_identifier: &str,
    arch_specific_definitions: &[&str],
    arglists: &str,
    passes: &str,
) -> String {
    format!(
        r#"{notices}{header_files}
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>

template<typename T1, typename T2> T1 cast(T2 x) {{
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret{{}};
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}}

std::ostream& operator<<(std::ostream& os, float16_t value) {{
    uint16_t temp = 0;
    memcpy(&temp, &value, sizeof(float16_t));
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(4) << std::hex << temp;
    os << ss.str();
    return os;
}}

#ifdef __{arch_identifier}__
{arch_specific_definitions}
#endif

{arglists}

int main(int argc, char **argv) {{
{passes}
    return 0;
}}"#,
        header_files = header_files
            .iter()
            .map(|header| format!("#include <{header}>"))
            .collect::<Vec<_>>()
            .join("\n"),
        arch_specific_definitions = arch_specific_definitions.into_iter().join("\n"),
    )
}

pub fn compile_c(compiler_commands: &[String]) -> bool {
    compiler_commands
        .par_iter()
        .map(|compiler_command| {
            let output = Command::new("sh").arg("-c").arg(compiler_command).output();
            if let Ok(output) = output {
                if output.status.success() {
                    true
                } else {
                    error!(
                        "Failed to compile code for intrinsics: \n\nstdout:\n{}\n\nstderr:\n{}",
                        std::str::from_utf8(&output.stdout).unwrap_or(""),
                        std::str::from_utf8(&output.stderr).unwrap_or("")
                    );
                    false
                }
            } else {
                error!("Command failed: {:#?}", output);
                false
            }
        })
        .find_any(|x| !x)
        .is_none()
}

pub fn create_c_files(identifiers: &Vec<String>) -> BTreeMap<&String, File> {
    identifiers
        .par_iter()
        .map(|identifier| {
            let c_filename = format!(r#"c_programs/{identifier}.cpp"#);
            let file = File::create(&c_filename).unwrap();

            (identifier, file)
        })
        .collect::<BTreeMap<&String, File>>()
}
