use serde::{Deserialize, Deserializer};

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
}

impl Argument {
    /// Creates an argument from a Rust style signature i.e. `name: type`
    fn from_rust(pos: usize, arg: &str) -> Result<Self, String> {
        let mut parts = arg.split(':');
        let name = parts.next().unwrap().trim().to_string();
        let ty = IntrinsicType::from_rust(parts.next().unwrap().trim())?;

        Ok(Self { pos, name, ty })
    }

    fn to_c_type(&self) -> String {
        self.ty.c_type()
    }

    fn is_simd(&self) -> bool {
        self.ty.is_simd()
    }

    pub fn is_ptr(&self) -> bool {
        self.ty.is_ptr()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList {
    pub args: Vec<Argument>,
}

impl ArgumentList {
    /// Creates an argument list from a Rust function signature, the data for
    /// this function should only be the arguments.
    /// e.g. for `fn test(a: u32, b: u32) -> u32` data should just be `a: u32, b: u32`
    fn from_rust_arguments(data: &str) -> Result<Self, String> {
        let args = data
            .split(',')
            .enumerate()
            .map(|(idx, arg)| Argument::from_rust(idx, arg))
            .collect::<Result<_, _>>()?;

        Ok(Self { args })
    }

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
            .map(|arg| arg.name.clone())
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Creates a line that initializes this argument for C code.
    /// e.g. `int32x2_t a = { 0x1, 0x2 };`
    pub fn init_random_values_c(&self, pass: usize) -> String {
        self.iter()
            .map(|arg| {
                format!(
                    "{ty} {name} = {{ {values} }};",
                    ty = arg.to_c_type(),
                    name = arg.name,
                    values = arg.ty.populate_random(pass, &Language::C)
                )
            })
            .collect::<Vec<_>>()
            .join("\n    ")
    }

    /// Creates a line that initializes this argument for Rust code.
    /// e.g. `let a = transmute([0x1, 0x2]);`
    pub fn init_random_values_rust(&self, pass: usize) -> String {
        self.iter()
            .map(|arg| {
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
            .collect::<Vec<_>>()
            .join("\n        ")
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument> {
        self.args.iter()
    }
}

impl<'de> Deserialize<'de> for ArgumentList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let s = String::deserialize(deserializer)?;
        Self::from_rust_arguments(&s).map_err(Error::custom)
    }
}
