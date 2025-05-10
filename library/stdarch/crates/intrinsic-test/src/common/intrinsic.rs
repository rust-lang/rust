use crate::common::argument::ArgumentList;
use crate::common::indentation::Indentation;
use crate::common::intrinsic_helpers::IntrinsicTypeDefinition;
use itertools::Itertools;

use super::argument::Argument;
use super::gen_c::generate_c_program;
use super::gen_rust::generate_rust_program;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic<T: IntrinsicTypeDefinition> {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList<T>,

    /// The return type of this intrinsic.
    pub results: T,

    /// Any architecture-specific tags.
    pub arch_tags: Vec<String>,
}

pub trait IntrinsicDefinition<T>
where
    T: IntrinsicTypeDefinition,
{
    fn arguments(&self) -> ArgumentList<T>;

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

    fn gen_code_c(
        &self,
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
                        pass = self.gen_code_c(
                            body_indentation,
                            constraints,
                            format!("{name}-{i}"),
                            target,
                        )
                    )
                })
                .join("\n")
        } else {
            self.generate_loop_c(indentation, &name, PASSES, target)
        }
    }

    fn generate_c_program(
        &self,
        header_files: &[&str],
        target: &str,
        notices: &str,
        arch_specific_definitions: &[&str],
    ) -> String {
        let arguments = self.arguments();
        let constraints = arguments
            .iter()
            .filter(|&i| i.has_constraint())
            .collect_vec();

        let indentation = Indentation::default();
        generate_c_program(
            notices,
            header_files,
            "aarch64",
            arch_specific_definitions,
            self.arguments()
                .gen_arglists_c(indentation, PASSES)
                .as_str(),
            self.gen_code_c(
                indentation.nested(),
                constraints.as_slice(),
                Default::default(),
                target,
            )
            .as_str(),
        )
    }

    fn gen_code_rust(
        &self,
        indentation: Indentation,
        constraints: &[&Argument<T>],
        name: String,
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
                            {body_indentation}const {name}: {ty} = {val};\n\
                            {pass}\n\
                        {indentation}}}",
                        name = current.name,
                        ty = current.ty.rust_type(),
                        val = i,
                        pass = self.gen_code_rust(
                            body_indentation,
                            constraints,
                            format!("{name}-{i}")
                        )
                    )
                })
                .join("\n")
        } else {
            self.generate_loop_rust(indentation, &name, PASSES)
        }
    }

    fn generate_rust_program(&self, target: &str, notice: &str, cfg: &str) -> String {
        let arguments = self.arguments();
        let constraints = arguments
            .iter()
            .filter(|i| i.has_constraint())
            .collect_vec();

        let indentation = Indentation::default();
        generate_rust_program(
            notice,
            cfg,
            target,
            self.arguments()
                .gen_arglists_rust(indentation.nested(), PASSES)
                .as_str(),
            self.gen_code_rust(indentation.nested(), &constraints, Default::default())
                .as_str(),
        )
    }
}
