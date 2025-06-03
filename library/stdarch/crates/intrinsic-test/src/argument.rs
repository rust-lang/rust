use std::ops::Range;

use crate::Language;
use crate::format::Indentation;
use crate::json_parser::ArgPrep;
use crate::types::{IntrinsicType, TypeKind};

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

impl TryFrom<ArgPrep> for Constraint {
    type Error = ();

    fn try_from(prep: ArgPrep) -> Result<Self, Self::Error> {
        let parsed_ints = match prep {
            ArgPrep::Immediate { min, max } => Ok((min, max)),
            _ => Err(()),
        };
        if let Ok((min, max)) = parsed_ints {
            if min == max {
                Ok(Constraint::Equal(min))
            } else {
                Ok(Constraint::Range(min..max + 1))
            }
        } else {
            Err(())
        }
    }
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

    pub fn type_and_name_from_c(arg: &str) -> (&str, &str) {
        let split_index = arg
            .rfind([' ', '*'])
            .expect("Couldn't split type and argname");

        (arg[..split_index + 1].trim_end(), &arg[split_index + 1..])
    }

    pub fn from_c(pos: usize, arg: &str, arg_prep: Option<ArgPrep>) -> Argument {
        let (ty, var_name) = Self::type_and_name_from_c(arg);

        let ty = IntrinsicType::from_c(ty)
            .unwrap_or_else(|_| panic!("Failed to parse argument '{arg}'"));

        let constraint = arg_prep.and_then(|a| a.try_into().ok());

        Argument {
            pos,
            name: String::from(var_name),
            ty,
            constraints: constraint.map_or(vec![], |r| vec![r]),
        }
    }

    fn is_rust_vals_array_const(&self) -> bool {
        use TypeKind::*;
        match self.ty {
            // Floats have to be loaded at runtime for stable NaN conversion.
            IntrinsicType::Type { kind: Float, .. } => false,
            IntrinsicType::Type {
                kind: Int | UInt | Poly,
                ..
            } => true,
            _ => unimplemented!(),
        }
    }

    /// The binding keyword (e.g. "const" or "let") for the array of possible test inputs.
    pub fn rust_vals_array_binding(&self) -> impl std::fmt::Display {
        if self.is_rust_vals_array_const() {
            "const"
        } else {
            "let"
        }
    }

    /// The name (e.g. "A_VALS" or "a_vals") for the array of possible test inputs.
    pub fn rust_vals_array_name(&self) -> impl std::fmt::Display {
        if self.is_rust_vals_array_const() {
            format!("{}_VALS", self.name.to_uppercase())
        } else {
            format!("{}_vals", self.name.to_lowercase())
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList {
    pub args: Vec<Argument>,
}

impl ArgumentList {
    /// Converts the argument list into the call parameters for a C function call.
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

    /// Converts the argument list into the call parameters for a Rust function.
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

    /// Creates a line for each argument that initializes an array for C from which `loads` argument
    /// values can be loaded  as a sliding window.
    /// e.g `const int32x2_t a_vals = {0x3effffff, 0x3effffff, 0x3f7fffff}`, if loads=2.
    pub fn gen_arglists_c(&self, indentation: Indentation, loads: u32) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    format!(
                        "{indentation}const {ty} {name}_vals[] = {values};",
                        ty = arg.ty.c_scalar_type(),
                        name = arg.name,
                        values = arg.ty.populate_random(indentation, loads, &Language::C)
                    )
                })
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Creates a line for each argument that initializes an array for Rust from which `loads` argument
    /// values can be loaded as a sliding window, e.g `const A_VALS: [u32; 20]  = [...];`
    pub fn gen_arglists_rust(&self, indentation: Indentation, loads: u32) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    format!(
                        "{indentation}{bind} {name}: [{ty}; {load_size}] = {values};",
                        bind = arg.rust_vals_array_binding(),
                        name = arg.rust_vals_array_name(),
                        ty = arg.ty.rust_scalar_type(),
                        load_size = arg.ty.num_lanes() * arg.ty.num_vectors() + loads - 1,
                        values = arg.ty.populate_random(indentation, loads, &Language::Rust)
                    )
                })
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Creates a line for each argument that initializes the argument from an array `[arg]_vals` at
    /// an offset `i` using a load intrinsic, in C.
    /// e.g `uint8x8_t a = vld1_u8(&a_vals[i]);`
    pub fn load_values_c(&self, indentation: Indentation, target: &str) -> String {
        self.iter()
            .filter_map(|arg| {
                // The ACLE doesn't support 64-bit polynomial loads on Armv7
                // This and the cast are a workaround for this
                let armv7_p64 = if let TypeKind::Poly = arg.ty.kind() {
                    target.contains("v7")
                } else {
                    false
                };

                (!arg.has_constraint()).then(|| {
                    format!(
                        "{indentation}{ty} {name} = {open_cast}{load}(&{name}_vals[i]){close_cast};\n",
                        ty = arg.to_c_type(),
                        name = arg.name,
                        load = if arg.is_simd() {
                            arg.ty.get_load_function(armv7_p64)
                        } else {
                            "*".to_string()
                        },
                        open_cast = if armv7_p64 {
                            format!("cast<{}>(", arg.to_c_type())
                        } else {
                            "".to_string()
                        },
                        close_cast = if armv7_p64 {
                            ")".to_string()
                        } else {
                            "".to_string()
                        }
                    )
                })
            })
            .collect()
    }

    /// Creates a line for each argument that initializes the argument from array `[ARG]_VALS` at
    /// an offset `i` using a load intrinsic, in Rust.
    /// e.g `let a = vld1_u8(A_VALS.as_ptr().offset(i));`
    pub fn load_values_rust(&self, indentation: Indentation) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    format!(
                        "{indentation}let {name} = {load}({vals_name}.as_ptr().offset(i));\n",
                        name = arg.name,
                        vals_name = arg.rust_vals_array_name(),
                        load = if arg.is_simd() {
                            arg.ty.get_load_function(false)
                        } else {
                            "*".to_string()
                        },
                    )
                })
            })
            .collect()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument> {
        self.args.iter()
    }
}
