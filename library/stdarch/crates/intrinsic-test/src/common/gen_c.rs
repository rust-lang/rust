use itertools::Itertools;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::process::Command;

use super::argument::Argument;
use super::indentation::Indentation;
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

// Formats the main C program template with placeholders
pub fn format_c_main_template(
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
        arch_specific_definitions = arch_specific_definitions.join("\n"),
    )
}

pub fn compile_c_programs(compiler_commands: &[String]) -> bool {
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

// Creates directory structure and file path mappings
pub fn setup_c_file_paths(identifiers: &Vec<String>) -> BTreeMap<&String, String> {
    let _ = std::fs::create_dir("c_programs");
    identifiers
        .par_iter()
        .map(|identifier| {
            let c_filename = format!(r#"c_programs/{identifier}.cpp"#);

            (identifier, c_filename)
        })
        .collect::<BTreeMap<&String, String>>()
}

pub fn generate_c_test_loop<T: IntrinsicTypeDefinition + Sized>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    additional: &str,
    passes: u32,
    _target: &str,
) -> String {
    let body_indentation = indentation.nested();
    format!(
        "{indentation}for (int i=0; i<{passes}; i++) {{\n\
            {loaded_args}\
            {body_indentation}auto __return_value = {intrinsic_call}({args});\n\
            {print_result}\n\
        {indentation}}}",
        loaded_args = intrinsic.arguments().load_values_c(body_indentation),
        intrinsic_call = intrinsic.name(),
        args = intrinsic.arguments().as_call_param_c(),
        print_result = intrinsic.print_result_c(body_indentation, additional)
    )
}

pub fn generate_c_constraint_blocks<T: IntrinsicTypeDefinition>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    constraints: &[&Argument<T>],
    name: String,
    target: &str,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraint
            .iter()
            .map(|c| c.to_range())
            .flat_map(|r| r.into_iter());

        let body_indentation = indentation.nested();
        range
            .map(|i| {
                format!(
                    "{indentation}{{\n\
                        {body_indentation}{ty} {name} = {val};\n\
                        {pass}\n\
                    {indentation}}}",
                    name = current.name,
                    ty = current.ty.c_type(),
                    val = i,
                    pass = generate_c_constraint_blocks(
                        intrinsic,
                        body_indentation,
                        constraints,
                        format!("{name}-{i}"),
                        target,
                    )
                )
            })
            .join("\n")
    } else {
        generate_c_test_loop(intrinsic, indentation, &name, PASSES, target)
    }
}

// Compiles C test programs using specified compiler
pub fn create_c_test_program<T: IntrinsicTypeDefinition>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    header_files: &[&str],
    target: &str,
    c_target: &str,
    notices: &str,
    arch_specific_definitions: &[&str],
) -> String {
    let arguments = intrinsic.arguments();
    let constraints = arguments
        .iter()
        .filter(|&i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    format_c_main_template(
        notices,
        header_files,
        c_target,
        arch_specific_definitions,
        intrinsic
            .arguments()
            .gen_arglists_c(indentation, PASSES)
            .as_str(),
        generate_c_constraint_blocks(
            intrinsic,
            indentation.nested(),
            constraints.as_slice(),
            Default::default(),
            target,
        )
        .as_str(),
    )
}
