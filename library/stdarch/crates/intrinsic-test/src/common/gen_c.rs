use itertools::Itertools;

use crate::common::constraint::Constraint;
use crate::common::intrinsic::Intrinsic;

use super::intrinsic_helpers::IntrinsicTypeDefinition;

fn generate_c_wrapper<'a, T: IntrinsicTypeDefinition + 'a>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
    constraints: &mut (impl Iterator<Item = &'a Constraint> + Clone),
    imm_values: &mut Vec<i64>,
) -> std::io::Result<()> {
    if let Some(current) = constraints.next() {
        for i in current.iter() {
            imm_values.push(i);
            generate_c_wrapper(w, intrinsic, &mut constraints.clone(), imm_values)?;
            imm_values.pop();
        }
    } else {
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
        )?;
    }
    Ok(())
}

fn create_c_wrapper<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
) -> std::io::Result<()> {
    generate_c_wrapper(
        w,
        intrinsic,
        &mut intrinsic
            .arguments
            .iter()
            .filter_map(|arg| arg.constraint.as_ref()),
        &mut Vec::new(),
    )
}

pub fn write_wrapper_c<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    notice: &str,
    platform_headers: &[&str],
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(w, "{notice}")?;

    for header in platform_headers {
        writeln!(w, "#include <{header}>")?;
    }

    for intrinsic in intrinsics {
        create_c_wrapper(w, intrinsic)?;
    }

    Ok(())
}
