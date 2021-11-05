use std::ops::Range;

use crate::types::IntrinsicType;
use crate::Language;

/// An argument for the intrinsic.
#[derive(Debug, PartialEq, Clone)]
pub struct Argument {
    /// The argument's index in the intrinsic function call.
    pub pos: usize,
    /// The argument name.
    pub name: String,
    /// The type of the argument.
    pub ty: IntrinsicType,
    /// Any constraints that are on this argument
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Constraint {
    Equal(i64),
    Range(Range<i64>),
}

impl Constraint {
    pub fn to_range(&self) -> Range<i64> {
        match self {
            Constraint::Equal(eq) => *eq..*eq + 1,
            Constraint::Range(range) => range.clone(),
        }
    }
}

impl Argument {
    fn to_c_type(&self) -> String {
        self.ty.c_type()
    }

    fn is_simd(&self) -> bool {
        self.ty.is_simd()
    }

    pub fn is_ptr(&self) -> bool {
        self.ty.is_ptr()
    }

    pub fn has_constraint(&self) -> bool {
        !self.constraints.is_empty()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList {
    pub args: Vec<Argument>,
}

impl ArgumentList {
    /// Converts the argument list into the call paramters for a C function call.
    /// e.g. this would generate something like `a, &b, c`
    pub fn as_call_param_c(&self) -> String {
        self.args
            .iter()
            .map(|arg| match arg.ty {
                IntrinsicType::Ptr { .. } => {
                    format!("&{}", arg.name)
                }
                IntrinsicType::Type { .. } => arg.name.clone(),
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Converts the argument list into the call paramters for a Rust function.
    /// e.g. this would generate something like `a, b, c`
    pub fn as_call_param_rust(&self) -> String {
        self.args
            .iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| arg.name.clone())
            .collect::<Vec<String>>()
            .join(", ")
    }

    pub fn as_constraint_parameters_rust(&self) -> String {
        self.args
            .iter()
            .filter(|a| a.has_constraint())
            .map(|arg| arg.name.clone())
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Creates a line that initializes this argument for C code.
    /// e.g. `int32x2_t a = { 0x1, 0x2 };`
    pub fn init_random_values_c(&self, pass: usize) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    format!(
                        "{ty} {name} = {{ {values} }};",
                        ty = arg.to_c_type(),
                        name = arg.name,
                        values = arg.ty.populate_random(pass, &Language::C)
                    )
                })
            })
            .collect::<Vec<_>>()
            .join("\n    ")
    }

    /// Creates a line that initializes this argument for Rust code.
    /// e.g. `let a = transmute([0x1, 0x2]);`
    pub fn init_random_values_rust(&self, pass: usize) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    if arg.is_simd() {
                        format!(
                            "let {name} = ::std::mem::transmute([{values}]);",
                            name = arg.name,
                            values = arg.ty.populate_random(pass, &Language::Rust),
                        )
                    } else {
                        format!(
                            "let {name} = {value};",
                            name = arg.name,
                            value = arg.ty.populate_random(pass, &Language::Rust)
                        )
                    }
                })
            })
            .collect::<Vec<_>>()
            .join("\n        ")
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument> {
        self.args.iter()
    }
}
