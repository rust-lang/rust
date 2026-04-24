use itertools::Itertools;

use crate::common::intrinsic_helpers::TypeKind;

use super::constraint::Constraint;
use super::gen_rust::PASSES;
use super::indentation::Indentation;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

/// An argument for the intrinsic.
#[derive(Debug, PartialEq, Clone)]
pub struct Argument<T: IntrinsicTypeDefinition> {
    /// The argument's index in the intrinsic function call.
    pub pos: usize,
    /// The argument name.
    pub name: String,
    /// The type of the argument.
    pub ty: T,
    /// Any constraints that are on this argument
    pub constraint: Option<Constraint>,
}

impl<T> Argument<T>
where
    T: IntrinsicTypeDefinition,
{
    pub fn new(pos: usize, name: String, ty: T, constraint: Option<Constraint>) -> Self {
        Argument {
            pos,
            name,
            ty,
            constraint,
        }
    }

    pub fn to_c_type(&self) -> String {
        self.ty.c_type()
    }

    pub fn generate_name(&self) -> String {
        format!("{}_val", self.name)
    }

    pub fn is_simd(&self) -> bool {
        self.ty.is_simd()
    }

    pub fn is_ptr(&self) -> bool {
        self.ty.is_ptr()
    }

    pub fn has_constraint(&self) -> bool {
        self.constraint.is_some()
    }

    /// The binding keyword (e.g. "const" or "let") for the array of possible test inputs.
    fn rust_vals_array_binding(&self) -> impl std::fmt::Display {
        if self.ty.is_rust_vals_array_const() {
            "const"
        } else {
            "let"
        }
    }

    /// The name (e.g. "A_VALS" or "a_vals") for the array of possible test inputs.
    pub(crate) fn rust_vals_array_name(&self) -> impl std::fmt::Display {
        if self.ty.is_rust_vals_array_const() {
            let loads = crate::common::gen_rust::PASSES;
            format!(
                "{}_{ty}_{load_size}",
                self.name.to_uppercase(),
                ty = self.ty.rust_scalar_type(),
                load_size = self.ty.num_lanes() * self.ty.num_vectors() + loads - 1,
            )
        } else {
            format!("{}_vals", self.name.to_lowercase())
        }
    }

    pub(crate) fn pass_by_ref(&self) -> bool {
        // pass SIMD types and `f16` by reference
        self.is_simd() || (self.ty.kind() == TypeKind::Float && self.ty.inner_size() == 16)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList<T: IntrinsicTypeDefinition> {
    pub args: Vec<Argument<T>>,
}

impl<T> ArgumentList<T>
where
    T: IntrinsicTypeDefinition,
{
    pub fn as_non_imm_arglist_c(&self) -> String {
        self.iter()
            .filter(|arg| !arg.has_constraint())
            .format_with("", |arg, fmt| {
                if arg.pass_by_ref() {
                    fmt(&format_args!(", const {}* {}", arg.to_c_type(), arg.name))
                } else {
                    fmt(&format_args!(", {} {}", arg.to_c_type(), arg.name))
                }
            })
            .to_string()
    }

    pub fn as_non_imm_arglist_rust(&self) -> String {
        self.iter()
            .filter(|arg| !arg.has_constraint())
            .format_with("", |arg, fmt| {
                if arg.pass_by_ref() {
                    fmt(&format_args!(
                        ", {}: *const {}",
                        arg.name,
                        arg.ty.rust_type()
                    ))
                } else {
                    fmt(&format_args!(", {}: {}", arg.name, arg.ty.rust_type()))
                }
            })
            .to_string()
    }

    pub fn as_call_params_c(&self, imm_args: &[i64]) -> String {
        let mut imm_args = imm_args.iter();
        self.iter()
            .format_with(", ", |arg, fmt| {
                if arg.has_constraint() {
                    fmt(&imm_args.next().unwrap())
                } else {
                    if arg.pass_by_ref() {
                        fmt(&"*")?;
                    }
                    fmt(&arg.name)
                }
            })
            .to_string()
    }

    /// Converts the argument list into the call parameters for a Rust function.
    /// e.g. this would generate something like `a, b, c`
    pub fn as_call_param_rust(&self) -> String {
        self.iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| arg.generate_name())
            .join(", ")
    }

    pub fn as_c_call_param_rust(&self) -> String {
        self.iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| {
                if arg.pass_by_ref() {
                    format!(", &raw const {}", arg.generate_name())
                } else {
                    format!(", {}", arg.generate_name())
                }
            })
            .join("")
    }

    /// Creates a line for each argument that initializes an array for Rust from which `loads` argument
    /// values can be loaded as a sliding window, e.g `const A_VALS: [u32; 20]  = [...];`
    pub fn gen_arglists_rust(
        &self,
        w: &mut impl std::io::Write,
        indentation: Indentation,
        loads: u32,
    ) -> std::io::Result<()> {
        for arg in self.iter().filter(|&arg| !arg.has_constraint()) {
            // Constants are defined globally.
            if arg.ty.is_rust_vals_array_const() {
                continue;
            }

            Self::gen_arg_rust(arg, w, indentation, loads)?;
        }

        Ok(())
    }

    pub fn gen_arg_rust(
        arg: &Argument<T>,
        w: &mut impl std::io::Write,
        indentation: Indentation,
        loads: u32,
    ) -> std::io::Result<()> {
        writeln!(
            w,
            "{indentation}{bind} {name}: [{ty}; {load_size}] = {values};\n",
            bind = arg.rust_vals_array_binding(),
            name = arg.rust_vals_array_name(),
            ty = arg.ty.rust_scalar_type(),
            load_size = arg.ty.num_lanes() * arg.ty.num_vectors() + loads - 1,
            values = arg.ty.populate_random(indentation, loads)
        )
    }

    /// Creates a line for each argument that initializes the argument from array `[ARG]_VALS` at
    /// an offset `i` using a load intrinsic, in Rust.
    /// e.g `let a = vld1_u8(A_VALS.as_ptr().offset(i));`
    pub fn load_values_rust(&self, indentation: Indentation) -> String {
        self.iter()
            .filter(|&arg| !arg.has_constraint())
            .enumerate()
            .map(|(idx, arg)| {
                if arg.is_simd() {
                    format!(
                        "{indentation}let {name} = {load}({vals_name}.as_ptr().add((i+{idx}) % {PASSES}) as _);\n",
                        name = arg.generate_name(),
                        vals_name = arg.rust_vals_array_name(),
                        load = arg.ty.get_load_function(),
                    )
                } else {
                    format!(
                        "{indentation}let {name} = {vals_name}[(i+{idx}) % {PASSES}];\n",
                        name = arg.generate_name(),
                        vals_name = arg.rust_vals_array_name(),
                    )
                }
            })
            .collect()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument<T>> {
        self.args.iter()
    }
}
