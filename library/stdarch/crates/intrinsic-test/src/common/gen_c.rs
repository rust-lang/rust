use crate::common::intrinsic::Intrinsic;

use super::argument::Argument;
use super::indentation::Indentation;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;
const COMMON_HEADERS: [&str; 7] = [
    "iostream",
    "string",
    "cstring",
    "iomanip",
    "sstream",
    "type_traits",
    "cassert",
];

pub fn generate_c_test_loop<T: IntrinsicTypeDefinition + Sized>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
    indentation: Indentation,
    additional: &str,
    passes: u32,
) -> std::io::Result<()> {
    let body_indentation = indentation.nested();
    writeln!(
        w,
        "{indentation}for (int i=0; i<{passes}; i++) {{\n\
            {loaded_args}\
            {body_indentation}auto __return_value = {intrinsic_call}({args});\n\
            {print_result}\n\
        {indentation}}}",
        loaded_args = intrinsic.arguments.load_values_c(body_indentation),
        intrinsic_call = intrinsic.name,
        args = intrinsic.arguments.as_call_param_c(),
        print_result = intrinsic
            .results
            .print_result_c(body_indentation, additional)
    )
}

pub fn generate_c_constraint_blocks<'a, T: IntrinsicTypeDefinition + 'a>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
    indentation: Indentation,
    constraints: &mut (impl Iterator<Item = &'a Argument<T>> + Clone),
    name: String,
) -> std::io::Result<()> {
    let Some(current) = constraints.next() else {
        return generate_c_test_loop(w, intrinsic, indentation, &name, PASSES);
    };

    let body_indentation = indentation.nested();
    for i in current.constraint.iter().flat_map(|c| c.iter()) {
        let ty = current.ty.c_type();

        writeln!(w, "{indentation}{{")?;

        // TODO: Move to actually specifying the enum value
        // instead of typecasting integers, for better clarity
        // of generated code.
        writeln!(
            w,
            "{body_indentation}const {ty} {} = ({ty}){i};",
            current.generate_name()
        )?;

        generate_c_constraint_blocks(
            w,
            intrinsic,
            body_indentation,
            &mut constraints.clone(),
            format!("{name}-{i}"),
        )?;

        writeln!(w, "{indentation}}}")?;
    }

    Ok(())
}

// Compiles C test programs using specified compiler
pub fn create_c_test_function<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
) -> std::io::Result<()> {
    let indentation = Indentation::default();

    writeln!(w, "int run_{}() {{", intrinsic.name)?;

    // Define the arrays of arguments.
    let arguments = &intrinsic.arguments;
    arguments.gen_arglists_c(w, indentation.nested(), PASSES)?;

    generate_c_constraint_blocks(
        w,
        intrinsic,
        indentation.nested(),
        &mut arguments.iter().rev().filter(|&i| i.has_constraint()),
        Default::default(),
    )?;

    writeln!(w, "    return 0;")?;
    writeln!(w, "}}")?;

    Ok(())
}

pub fn write_mod_cpp<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    notice: &str,
    platform_headers: &[&str],
    forward_declarations: &str,
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(w, "{notice}")?;

    for header in COMMON_HEADERS.iter().chain(platform_headers.iter()) {
        writeln!(w, "#include <{header}>")?;
    }

    writeln!(w, "{}", forward_declarations)?;

    for intrinsic in intrinsics {
        create_c_test_function(w, intrinsic)?;
    }

    Ok(())
}

pub fn write_main_cpp<'a>(
    w: &mut impl std::io::Write,
    arch_specific_definitions: &str,
    arch_specific_headers: &[&str],
    intrinsics: impl Iterator<Item = &'a str> + Clone,
) -> std::io::Result<()> {
    for header in COMMON_HEADERS.iter().chain(arch_specific_headers.iter()) {
        writeln!(w, "#include <{header}>")?;
    }

    // NOTE: It's assumed that this value contains the required `ifdef`s.
    writeln!(w, "{arch_specific_definitions }")?;

    for intrinsic in intrinsics.clone() {
        writeln!(w, "extern int run_{intrinsic}(void);")?;
    }

    writeln!(w, "int main(int argc, char **argv) {{")?;
    writeln!(w, "    std::string intrinsic_name = argv[1];")?;

    writeln!(w, "    if (false) {{")?;

    for intrinsic in intrinsics {
        writeln!(w, "    }} else if (intrinsic_name == \"{intrinsic}\") {{")?;
        writeln!(w, "        return run_{intrinsic}();")?;
    }

    writeln!(w, "    }} else {{")?;
    writeln!(
        w,
        "        std::cerr << \"Unknown command: \" << intrinsic_name << \"\\n\";"
    )?;
    writeln!(w, "        return -1;")?;
    writeln!(w, "    }}")?;

    writeln!(w, "}}")?;

    Ok(())
}
