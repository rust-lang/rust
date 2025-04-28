use crate::base_intrinsictype_trait_def_macro;
use crate::common::argument::ArgumentList;
use crate::common::cli::Language;
use crate::common::indentation::Indentation;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::{
    BaseIntrinsicTypeDefinition, IntrinsicTypeDefinition, TypeKind,
};

base_intrinsictype_trait_def_macro! {ArmIntrinsicType}

impl IntrinsicDefinition<ArmIntrinsicType> for Intrinsic<ArmIntrinsicType> {
    fn arguments(&self) -> ArgumentList<ArmIntrinsicType> {
        self.arguments.clone()
    }

    fn results(&self) -> ArmIntrinsicType {
        self.results.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, indentation: Indentation, additional: &str) -> String {
        let lanes = if self.results().num_vectors() > 1 {
            (0..self.results().num_vectors())
                .map(|vector| {
                    format!(
                        r#""{ty}(" << {lanes} << ")""#,
                        ty = self.results().c_single_vector_type(),
                        lanes = (0..self.results().num_lanes())
                            .map(move |idx| -> std::string::String {
                                format!(
                                    "{cast}{lane_fn}(__return_value.val[{vector}], {lane})",
                                    cast = self.results().c_promotion(),
                                    lane_fn = self.results().get_lane_function(),
                                    lane = idx,
                                    vector = vector,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(r#" << ", " << "#)
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else if self.results().num_lanes() > 1 {
            (0..self.results().num_lanes())
                .map(|idx| -> std::string::String {
                    format!(
                        "{cast}{lane_fn}(__return_value, {lane})",
                        cast = self.results().c_promotion(),
                        lane_fn = self.results().get_lane_function(),
                        lane = idx
                    )
                })
                .collect::<Vec<_>>()
                .join(r#" << ", " << "#)
        } else {
            format!(
                "{promote}cast<{cast}>(__return_value)",
                cast = match self.results.kind() {
                    TypeKind::Float if self.results().inner_size() == 16 => "float16_t".to_string(),
                    TypeKind::Float if self.results().inner_size() == 32 => "float".to_string(),
                    TypeKind::Float if self.results().inner_size() == 64 => "double".to_string(),
                    TypeKind::Int => format!("int{}_t", self.results().inner_size()),
                    TypeKind::UInt => format!("uint{}_t", self.results().inner_size()),
                    TypeKind::Poly => format!("poly{}_t", self.results().inner_size()),
                    ty => todo!("print_result_c - Unknown type: {:#?}", ty),
                },
                promote = self.results().c_promotion(),
            )
        };

        format!(
            r#"{indentation}std::cout << "Result {additional}-" << i+1 << ": {ty}" << std::fixed << std::setprecision(150) <<  {lanes} << "{close}" << std::endl;"#,
            ty = if self.results().is_simd() {
                format!("{}(", self.results().c_type())
            } else {
                String::from("")
            },
            close = if self.results.is_simd() { ")" } else { "" },
        )
    }

    pub fn generate_loop_c(
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
            loaded_args = self.arguments.load_values_c(body_indentation),
            intrinsic_call = self.name,
            args = self.arguments.as_call_param_c(),
            print_result = self.print_result_c(body_indentation, additional)
        )
    }

    pub fn generate_loop_rust(
        &self,
        indentation: Indentation,
        additional: &str,
        passes: u32,
    ) -> String {
        let constraints = self.arguments.as_constraint_parameters_rust();
        let constraints = if !constraints.is_empty() {
            format!("::<{constraints}>")
        } else {
            constraints
        };

        // the `intrinsic-test` crate compares the output of C and Rust intrinsics. Currently, It uses
        // a string representation of the output value to compare. In C, f16 values are currently printed
        // as hexadecimal integers. Since https://github.com/rust-lang/rust/pull/127013, rust does print
        // them as decimal floating point values. To keep the intrinsics tests working, for now, format
        // vectors containing f16 values like C prints them.
        let return_value = match self.results.kind() {
            TypeKind::Float if self.results.inner_size() == 16 => "debug_f16(__return_value)",
            _ => "format_args!(\"{__return_value:.150?}\")",
        };

        let indentation2 = indentation.nested();
        let indentation3 = indentation2.nested();

        format!(
            "{indentation}for i in 0..{passes} {{\n\
                {indentation2}unsafe {{\n\
                    {loaded_args}\
                    {indentation3}let __return_value = {intrinsic_call}{const}({args});\n\
                    {indentation3}println!(\"Result {additional}-{{}}: {{:?}}\", i + 1, {return_value});\n\
                {indentation2}}}\n\
            {indentation}}}",
            loaded_args = self.arguments.load_values_rust(indentation3),
            intrinsic_call = self.name,
            const = constraints,
            args = self.arguments.as_call_param_rust(),
        )
    }
}
