use crate::common::argument::ArgumentList;
use crate::common::format::Indentation;
use crate::common::intrinsic_types::IntrinsicTypeDefinition;

use super::argument::MetadataDefinition;

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic<T: IntrinsicTypeDefinition, M: MetadataDefinition> {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList<T, M>,

    /// The return type of this intrinsic.
    pub results: T,

    /// Any architecture-specific tags.
    pub arch_tags: Vec<String>,
}

pub trait IntrinsicDefinition<T, M>
where
    T: IntrinsicTypeDefinition,
    M: MetadataDefinition,
{
    fn arguments(&self) -> ArgumentList<T, M>;

    fn results(&self) -> T;

    fn name(&self) -> String;

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, _indentation: Indentation, _additional: &str) -> String {
        unimplemented!("Architectures need to implement print_result_c!")
    }

    fn generate_loop_c(
        &self,
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
            loaded_args = self.arguments().load_values_c(body_indentation),
            intrinsic_call = self.name(),
            args = self.arguments().as_call_param_c(),
            print_result = self.print_result_c(body_indentation, additional)
        )
    }

    fn generate_loop_rust(
        &self,
        indentation: Indentation,
        additional: &str,
        passes: u32,
    ) -> String {
        let constraints = self.arguments().as_constraint_parameters_rust();
        let constraints = if !constraints.is_empty() {
            format!("::<{constraints}>")
        } else {
            constraints
        };

        let indentation2 = indentation.nested();
        let indentation3 = indentation2.nested();
        format!(
            "{indentation}for i in 0..{passes} {{\n\
                {indentation2}unsafe {{\n\
                    {loaded_args}\
                    {indentation3}let __return_value = {intrinsic_call}{const}({args});\n\
                    {indentation3}println!(\"Result {additional}-{{}}: {{:.150?}}\", i + 1, __return_value);\n\
                {indentation2}}}\n\
            {indentation}}}",
            loaded_args = self.arguments().load_values_rust(indentation3),
            intrinsic_call = self.name(),
            const = constraints,
            args = self.arguments().as_call_param_rust(),
        )
    }
}
