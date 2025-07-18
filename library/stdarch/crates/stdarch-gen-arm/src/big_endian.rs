use crate::expression::LetVariant;
use crate::wildstring::WildStringPart;
use crate::{
    expression::{Expression, IdentifierType},
    typekinds::*,
    wildstring::WildString,
};

/// Simplifies creating a string that can be used in an Expression, as Expression
/// expects all strings to be `WildString`
fn create_single_wild_string(name: &str) -> WildString {
    WildString(vec![WildStringPart::String(name.to_string())])
}

/// Creates an Identifier with name `name` with no wildcards. This, for example,
/// can be used to create variables, function names or arbitrary input. Is is
/// extremely flexible.
pub fn create_symbol_identifier(arbitrary_string: &str) -> Expression {
    let identifier_name = create_single_wild_string(arbitrary_string);
    Expression::Identifier(identifier_name, IdentifierType::Symbol)
}

/// To compose the simd_shuffle! call we need:
/// - simd_shuffle!(<arg1>, <arg2>, <array>)
///
/// Here we are creating a string version of the `<array>` that can be used as an
/// Expression Identifier
///
/// In textual form `a: int32x4_t` which has 4 lanes would generate:
/// ```
/// [0, 1, 2, 3]
/// ```
fn create_array(lanes: u32) -> Option<String> {
    match lanes {
        1 => None, /* Makes no sense to shuffle an array of size 1 */
        2 => Some("[1, 0]".to_string()),
        3 => Some("[2, 1, 0]".to_string()),
        4 => Some("[3, 2, 1, 0]".to_string()),
        8 => Some("[7, 6, 5, 4, 3, 2, 1, 0]".to_string()),
        16 => Some("[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]".to_string()),
        _ => panic!("Incorrect vector number of vector lanes: {lanes}"),
    }
}

/// Creates: `let <variable_name>: <type> = <expression>`
pub fn create_let_variable(
    variable_name: &str,
    type_kind: &TypeKind,
    expression: Expression,
) -> Expression {
    let identifier_name = create_single_wild_string(variable_name);
    Expression::Let(LetVariant::WithType(
        identifier_name,
        type_kind.clone(),
        Box::new(expression),
    ))
}

pub fn create_mut_let_variable(
    variable_name: &str,
    type_kind: &TypeKind,
    expression: Expression,
) -> Expression {
    let identifier_name = create_single_wild_string(variable_name);
    Expression::Let(LetVariant::MutWithType(
        identifier_name,
        type_kind.clone(),
        Box::new(expression),
    ))
}

pub fn type_has_tuple(type_kind: &TypeKind) -> bool {
    if let TypeKind::Vector(vector_type) = type_kind {
        vector_type.tuple_size().is_some()
    } else {
        false
    }
}

pub fn make_variable_mutable(variable_name: &str, type_kind: &TypeKind) -> Expression {
    let mut_variable = format!("let mut {variable_name}: {type_kind} = {variable_name}");
    let identifier_name = create_single_wild_string(&mut_variable);
    Expression::Identifier(identifier_name, IdentifierType::Symbol)
}

/// For creating shuffle calls, accepts function pointers for formatting for tuple
/// types and types without a tuple
///
/// Example:
///
/// `a: int32x4_t` with formatting function `create_shuffle_call_fmt` creates:
/// ```
/// simd_shuffle!(a, a, [0, 1, 2, 3])
/// ```
///
/// `a: int32x4x2_t` creates:
/// ```
/// a.0 = simd_shuffle!(a.0, a.0, [0, 1, 2, 3])
/// a.1 = simd_shuffle!(a.1, a.1, [0, 1, 2, 3])
/// ```
fn create_shuffle_internal(
    variable_name: &String,
    type_kind: &TypeKind,
    fmt_tuple: fn(variable_name: &String, idx: u32, array_lanes: &String) -> String,
    fmt: fn(variable_name: &String, type_kind: &TypeKind, array_lanes: &String) -> String,
) -> Option<Expression> {
    let TypeKind::Vector(vector_type) = type_kind else {
        return None;
    };

    let lane_count = vector_type.lanes();
    let array_lanes = create_array(lane_count)?;

    let tuple_count = vector_type.tuple_size().map_or_else(|| 0, |t| t.to_int());

    if tuple_count > 0 {
        let capacity_estimate: usize =
            tuple_count as usize * (lane_count as usize + ((variable_name.len() + 2) * 3));
        let mut string_builder = String::with_capacity(capacity_estimate);

        /* <var_name>.idx = simd_shuffle!(<var_name>.idx, <var_name>.idx, [<indexes>]) */
        for idx in 0..tuple_count {
            let formatted = fmt_tuple(variable_name, idx, &array_lanes);
            string_builder += formatted.as_str();
        }
        Some(create_symbol_identifier(&string_builder))
    } else {
        /* Generate a list of shuffles for each tuple */
        let expression = fmt(variable_name, type_kind, &array_lanes);
        Some(create_symbol_identifier(&expression))
    }
}

fn create_assigned_tuple_shuffle_call_fmt(
    variable_name: &String,
    idx: u32,
    array_lanes: &String,
) -> String {
    format!(
        "{variable_name}.{idx} = unsafe {{ simd_shuffle!({variable_name}.{idx}, {variable_name}.{idx}, {array_lanes}) }};\n"
    )
}

fn create_assigned_shuffle_call_fmt(
    variable_name: &String,
    type_kind: &TypeKind,
    array_lanes: &String,
) -> String {
    format!(
        "let {variable_name}: {type_kind} = unsafe {{ simd_shuffle!({variable_name}, {variable_name}, {array_lanes}) }}"
    )
}

fn create_shuffle_call_fmt(
    variable_name: &String,
    _type_kind: &TypeKind,
    array_lanes: &String,
) -> String {
    format!("simd_shuffle!({variable_name}, {variable_name}, {array_lanes})")
}

/// Create a `simd_shuffle!(<...>, [...])` call, where the output is stored
/// in a variable named `variable_name`
pub fn create_assigned_shuffle_call(
    variable_name: &String,
    type_kind: &TypeKind,
) -> Option<Expression> {
    create_shuffle_internal(
        variable_name,
        type_kind,
        create_assigned_tuple_shuffle_call_fmt,
        create_assigned_shuffle_call_fmt,
    )
}

/// Create a `simd_shuffle!(<...>, [...])` call
pub fn create_shuffle_call(variable_name: &String, type_kind: &TypeKind) -> Option<Expression> {
    create_shuffle_internal(
        variable_name,
        type_kind,
        create_assigned_tuple_shuffle_call_fmt,
        create_shuffle_call_fmt,
    )
}
