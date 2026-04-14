use crate::ir;
use std::collections::HashSet;

pub fn validate(schema: &ir::Schema) -> Result<(), String> {
    for kind in schema.kinds.values() {
        if let ir::KindShape::Struct(fields) = &kind.shape {
            let mut visited = HashSet::new();
            check_recursive_struct(kind.canonical_name.as_str(), fields, schema, &mut visited)?;
        }
    }
    Ok(())
}

fn check_recursive_struct(
    root_name: &str,
    fields: &[ir::ResolvedField],
    schema: &ir::Schema,
    visited: &mut HashSet<String>,
) -> Result<(), String> {
    for f in fields {
        check_type_recursion(root_name, &f.ty, schema, visited)?;
    }
    Ok(())
}

fn check_type_recursion(
    root_name: &str,
    ty: &ir::ResolvedType,
    schema: &ir::Schema,
    visited: &mut HashSet<String>,
) -> Result<(), String> {
    // ref<T> breaks recursion
    if ty.kind_ref == "ref" {
        return Ok(());
    }

    if ty.kind_ref == root_name {
        return Err(format!(
            "Direct recursion detected in struct {}: field refers to itself without ref<>",
            root_name
        ));
    }

    if visited.contains(&ty.kind_ref) {
        return Ok(());
    }

    visited.insert(ty.kind_ref.clone());

    if let Some(kind) = schema.kinds.get(&ty.kind_ref) {
        if let ir::KindShape::Struct(fields) = &kind.shape {
            check_recursive_struct(root_name, fields, schema, visited)?;
        }
    }

    for arg in &ty.args {
        check_type_recursion(root_name, arg, schema, visited)?;
    }

    Ok(())
}
