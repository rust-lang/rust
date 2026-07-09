use itertools::Itertools;

use crate::common::{SupportedArchitecture, intrinsic::Intrinsic};

use super::intrinsic_helpers::TypeDefinition;

/// Generates a C source file containing wrapper functions around each specialisation of each
/// intrinsic (that is, intrinsics with specific values for the the immediate arguments). Each
/// wrapper function is invoked via FFI from the Rust binary doing the testing.
///
/// e.g.
/// ```c
/// void __crc32cd_wrapper(uint32_t* __dst, uint32_t a, uint64_t b) {
///    *__dst = __crc32cd(a, b);
/// }
/// ```
pub fn write_wrapper_c<A: SupportedArchitecture>(
    w: &mut impl std::io::Write,
    intrinsics: &[Intrinsic<A>],
) -> std::io::Result<()> {
    write!(
        w,
        r#"
{notice}
#include <stdint.h>
#include <stddef.h>
{prelude}

{intrinsics}
"#,
        notice = A::NOTICE,
        prelude = A::C_PRELUDE,
        intrinsics = intrinsics.iter().format_with("", |intrinsic, fmt| {
            fmt(&intrinsic
                .specializations()
                .format_with("\n", |imm_values, fmt| {
                    fmt(&format_args!(
                        "
void {name}_wrapper{imm_arglist}({return_ty}* __dst{arglist}) {{
    *__dst = {name}({params});
}}",
                        return_ty = intrinsic.results.c_type(),
                        name = intrinsic.name,
                        imm_arglist = imm_values
                            .iter()
                            .format_with("", |i, fmt| fmt(&format_args!("_{i}"))),
                        arglist = intrinsic.arguments.as_non_imm_arglist_c(),
                        params = intrinsic.arguments.as_call_params_c(&imm_values)
                    ))
                }))
        }),
    )
}
