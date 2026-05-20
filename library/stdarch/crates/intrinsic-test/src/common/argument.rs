use itertools::Itertools;

use crate::common::intrinsic_helpers::TypeKind;
use crate::common::values::{test_values_array, test_values_array_length};

use super::constraint::Constraint;
use super::gen_rust::PASSES;
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

    /// Returns a string with the name of the static variable containing test values for intrinsic
    /// arguments of this type.
    pub(crate) fn rust_vals_array_name(&self) -> impl std::fmt::Display {
        let loads = crate::common::gen_rust::PASSES;
        format!(
            "{ty}_{load_size}",
            ty = self.ty.rust_scalar_type().to_uppercase(),
            load_size = test_values_array_length(&self.ty, loads),
        )
    }

    /// Should this argument be passed by reference in C wrapper function declarations?
    ///
    /// SIMD types and `f16` are currently passed by reference.
    pub(crate) fn pass_by_ref(&self) -> bool {
        self.is_simd() || (self.ty.kind() == TypeKind::Float && self.ty.inner_size() == 16)
    }
}

/// Arguments of an intrinsic - including parameters that end up being const generics.
#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList<T: IntrinsicTypeDefinition> {
    pub args: Vec<Argument<T>>,
}

impl<T> ArgumentList<T>
where
    T: IntrinsicTypeDefinition,
{
    /// Returns a string with the arguments in `self` as a parameter list for a wrapper fn
    /// definition in C (e.g. `$ty1 $arg1, $ty2 $arg2`).
    ///
    /// Skips arguments with constraints - which correspond to arguments that must take immediates -
    /// as a different C definition will be generated for each value of these being tested.
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

    /// Returns a string with the arguments in `self` as a parameter list for a Rust declaration of
    /// a C wrapper fn (e.g. `$arg1: $ty1, $arg2: $ty2`).
    ///
    /// Skips arguments with constraints - which correspond to arguments that must take immediates -
    /// as a different C definition will be generated for each value of these being tested.
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

    /// Returns a string with the arguments in `self` being passed to an intrinsic call in C
    /// (e.g. `$arg1, 2 /* imm_args[0] */, $arg3` where `$arg2` has a constraint).
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

    /// Returns a string with the arguments in `self` being passed to an intrinsic call in Rust.
    /// (e.g. `$arg1, $arg3` where `$arg2` has a constraint and so corresponds to a const generic
    /// parameter).
    pub fn as_call_param_rust(&self) -> String {
        self.iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| arg.generate_name())
            .join(", ")
    }

    /// Returns a string with the arguments in `self` being passed to the declaration of a C wrapper
    /// fn from Rust (e.g. `$arg1, $arg3` (where `$arg2` has a constraint and so corresponds to a
    /// const generic parameter).
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

    /// Returns a string defining a static variable with test values used for all intrinsics with
    /// arguments of `arg`'s type.
    ///
    /// e.g.
    /// ```rust,ignore
    /// static U8_20: [u8; 20] = [
    ///     0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0xf0,
    ///     0x80, 0x3b, 0xff,
    /// ];
    /// ```
    pub fn gen_arg_rust(
        arg: &Argument<T>,
        w: &mut impl std::io::Write,
        loads: u32,
    ) -> std::io::Result<()> {
        writeln!(
            w,
            "static {name}: [{ty}; {load_size}] = {values};\n",
            name = arg.rust_vals_array_name(),
            ty = arg.ty.rust_scalar_type(),
            load_size = test_values_array_length(&arg.ty, loads),
            values = test_values_array(&arg.ty, loads)
        )
    }

    /// Returns a string defining a local variable for each argument and loading a value into each
    /// using a load intrinsic.
    ///
    /// e.g.
    /// ```rust,ignore
    /// let a = vld1_u8(I16_23.as_ptr().offset((i + 0 /* idx */) % 20 /* PASSES */));
    /// ````
    ///
    /// The generator will have already generated arrays of appropriate length with values that can
    /// be used for testing (see the `gen_args_rust` function).
    ///
    /// Each load is assumed to have a variable `i` in scope which comes from a loop which repeats
    /// the testing of the intrinsic for different values - each subsequent `i` shifts the window
    /// of values being loaded along the pre-prepared array.
    ///
    /// Each subsequent argument's first window is started one element further into the array
    /// then the previous.
    pub fn load_values_rust(&self) -> String {
        self.iter()
            .filter(|&arg| !arg.has_constraint())
            .enumerate()
            .map(|(idx, arg)| {
                if arg.is_simd() {
                    format!(
                        "let {name} = {load}({vals_name}.as_ptr().add((i+{idx}) % {PASSES}) as _);\n",
                        name = arg.generate_name(),
                        vals_name = arg.rust_vals_array_name(),
                        load = arg.ty.get_load_function(),
                    )
                } else {
                    format!(
                        "let {name} = {vals_name}[(i+{idx}) % {PASSES}];\n",
                        name = arg.generate_name(),
                        vals_name = arg.rust_vals_array_name(),
                    )
                }
            })
            .collect()
    }

    /// Returns an iterator over the contained arguments
    pub fn iter(&self) -> std::slice::Iter<'_, Argument<T>> {
        self.args.iter()
    }
}
