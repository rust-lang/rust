use itertools::Itertools;

use crate::common::intrinsic::Intrinsic;

use super::intrinsic_helpers::IntrinsicTypeDefinition;

pub fn write_wrapper_c<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    notice: &str,
    platform_headers: &[&str],
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(w, "{notice}")?;

    writeln!(w, "#include <stdint.h>")?;
    writeln!(w, "#include <stddef.h>")?;

    for header in platform_headers {
        writeln!(w, "#include <{header}>")?;
    }

    for intrinsic in intrinsics {
        intrinsic.iter_specializations(|imm_values| {
            writeln!(
                w,
                "
{return_ty} {name}_wrapper{imm_arglist}({arglist}) {{
    return {name}({params});
}}",
                return_ty = intrinsic.results.c_type(),
                name = intrinsic.name,
                imm_arglist = imm_values
                    .iter()
                    .format_with("", |i, fmt| fmt(&format_args!("_{i}"))),
                arglist = intrinsic.arguments.as_non_imm_arglist_c(),
                params = intrinsic.arguments.as_call_params_c(&imm_values)
            )
        })?;
    }

    Ok(())
}
