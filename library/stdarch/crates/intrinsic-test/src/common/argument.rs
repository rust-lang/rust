use super::cli::Language;
use super::constraint::Constraint;
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
    fn rust_vals_array_name(&self) -> impl std::fmt::Display {
        if self.ty.is_rust_vals_array_const() {
            format!("{}_VALS", self.name.to_uppercase())
        } else {
            format!("{}_vals", self.name.to_lowercase())
        }
    }

    fn as_call_param_c(&self) -> String {
        self.ty.as_call_param_c(&self.name)
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
    /// Converts the argument list into the call parameters for a C function call.
    /// e.g. this would generate something like `a, &b, c`
    pub fn as_call_param_c(&self) -> String {
        self.iter()
            .map(|arg| arg.as_call_param_c())
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Converts the argument list into the call parameters for a Rust function.
    /// e.g. this would generate something like `a, b, c`
    pub fn as_call_param_rust(&self) -> String {
        self.iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| arg.name.clone())
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Creates a line for each argument that initializes an array for C from which `loads` argument
    /// values can be loaded  as a sliding window.
    /// e.g `const int32x2_t a_vals = {0x3effffff, 0x3effffff, 0x3f7fffff}`, if loads=2.
    pub fn gen_arglists_c(
        &self,
        w: &mut impl std::io::Write,
        indentation: Indentation,
        loads: u32,
    ) -> std::io::Result<()> {
        for arg in self.iter().filter(|&arg| !arg.has_constraint()) {
            writeln!(
                w,
                "{indentation}const {ty} {name}_vals[] = {values};",
                ty = arg.ty.c_scalar_type(),
                name = arg.name,
                values = arg.ty.populate_random(indentation, loads, &Language::C)
            )?
        }

        Ok(())
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
            writeln!(
                w,
                "{indentation}{bind} {name}: [{ty}; {load_size}] = {values};",
                bind = arg.rust_vals_array_binding(),
                name = arg.rust_vals_array_name(),
                ty = arg.ty.rust_scalar_type(),
                load_size = arg.ty.num_lanes() * arg.ty.num_vectors() + loads - 1,
                values = arg.ty.populate_random(indentation, loads, &Language::Rust)
            )?
        }

        Ok(())
    }

    /// Creates a line for each argument that initializes the argument from an array `[arg]_vals` at
    /// an offset `i` using a load intrinsic, in C.
    /// e.g `uint8x8_t a = vld1_u8(&a_vals[i]);`
    ///
    /// ARM-specific
    pub fn load_values_c(&self, indentation: Indentation) -> String {
        self.iter()
            .filter(|&arg| !arg.has_constraint())
            .map(|arg| {
                format!(
                    "{indentation}{ty} {name} = cast<{ty}>({load}(&{name}_vals[i]));\n",
                    ty = arg.to_c_type(),
                    name = arg.name,
                    load = if arg.is_simd() {
                        arg.ty.get_load_function(Language::C)
                    } else {
                        "*".to_string()
                    }
                )
            })
            .collect()
    }

    /// Creates a line for each argument that initializes the argument from array `[ARG]_VALS` at
    /// an offset `i` using a load intrinsic, in Rust.
    /// e.g `let a = vld1_u8(A_VALS.as_ptr().offset(i));`
    pub fn load_values_rust(&self, indentation: Indentation) -> String {
        self.iter()
            .filter(|&arg| !arg.has_constraint())
            .map(|arg| {
                format!(
                    "{indentation}let {name} = {load}({vals_name}.as_ptr().offset(i));\n",
                    name = arg.name,
                    vals_name = arg.rust_vals_array_name(),
                    load = if arg.is_simd() {
                        arg.ty.get_load_function(Language::Rust)
                    } else {
                        "*".to_string()
                    },
                )
            })
            .collect()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument<T>> {
        self.args.iter()
    }
}
