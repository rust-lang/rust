//! Display and pretty printing routines.

use std::{
    fmt::{self, Write},
    mem,
};

use hir_expand::mod_path::PathKind;
use itertools::Itertools;
use span::Edition;

use crate::{
    db::DefDatabase,
    lang_item::LangItemTarget,
    path::{GenericArg, GenericArgs, Path},
    type_ref::{
        Mutability, TraitBoundModifier, TypeBound, TypeRef, TypeRefId, TypesMap, UseArgRef,
    },
};

pub(crate) fn print_path(
    db: &dyn DefDatabase,
    path: &Path,
    map: &TypesMap,
    buf: &mut dyn Write,
    edition: Edition,
) -> fmt::Result {
    if let Path::LangItem(it, s) = path {
        write!(buf, "builtin#lang(")?;
        match *it {
            LangItemTarget::ImplDef(it) => write!(buf, "{it:?}")?,
            LangItemTarget::EnumId(it) => {
                write!(buf, "{}", db.enum_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::Function(it) => {
                write!(buf, "{}", db.function_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::Static(it) => {
                write!(buf, "{}", db.static_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::Struct(it) => {
                write!(buf, "{}", db.struct_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::Union(it) => {
                write!(buf, "{}", db.union_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::TypeAlias(it) => {
                write!(buf, "{}", db.type_alias_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::Trait(it) => {
                write!(buf, "{}", db.trait_data(it).name.display(db.upcast(), edition))?
            }
            LangItemTarget::EnumVariant(it) => {
                write!(buf, "{}", db.enum_variant_data(it).name.display(db.upcast(), edition))?
            }
        }

        if let Some(s) = s {
            write!(buf, "::{}", s.display(db.upcast(), edition))?;
        }
        return write!(buf, ")");
    }
    match path.type_anchor() {
        Some(anchor) => {
            write!(buf, "<")?;
            print_type_ref(db, anchor, map, buf, edition)?;
            write!(buf, ">::")?;
        }
        None => match path.kind() {
            PathKind::Plain => {}
            &PathKind::SELF => write!(buf, "self")?,
            PathKind::Super(n) => {
                for i in 0..*n {
                    if i == 0 {
                        buf.write_str("super")?;
                    } else {
                        buf.write_str("::super")?;
                    }
                }
            }
            PathKind::Crate => write!(buf, "crate")?,
            PathKind::Abs => {}
            PathKind::DollarCrate(_) => write!(buf, "$crate")?,
        },
    }

    for (i, segment) in path.segments().iter().enumerate() {
        if i != 0 || !matches!(path.kind(), PathKind::Plain) {
            write!(buf, "::")?;
        }

        write!(buf, "{}", segment.name.display(db.upcast(), edition))?;
        if let Some(generics) = segment.args_and_bindings {
            write!(buf, "::<")?;
            print_generic_args(db, generics, map, buf, edition)?;

            write!(buf, ">")?;
        }
    }

    Ok(())
}

pub(crate) fn print_generic_args(
    db: &dyn DefDatabase,
    generics: &GenericArgs,
    map: &TypesMap,
    buf: &mut dyn Write,
    edition: Edition,
) -> fmt::Result {
    let mut first = true;
    let args = if generics.has_self_type {
        let (self_ty, args) = generics.args.split_first().unwrap();
        write!(buf, "Self=")?;
        print_generic_arg(db, self_ty, map, buf, edition)?;
        first = false;
        args
    } else {
        &generics.args
    };
    for arg in args {
        if !first {
            write!(buf, ", ")?;
        }
        first = false;
        print_generic_arg(db, arg, map, buf, edition)?;
    }
    for binding in generics.bindings.iter() {
        if !first {
            write!(buf, ", ")?;
        }
        first = false;
        write!(buf, "{}", binding.name.display(db.upcast(), edition))?;
        if !binding.bounds.is_empty() {
            write!(buf, ": ")?;
            print_type_bounds(db, &binding.bounds, map, buf, edition)?;
        }
        if let Some(ty) = binding.type_ref {
            write!(buf, " = ")?;
            print_type_ref(db, ty, map, buf, edition)?;
        }
    }
    Ok(())
}

pub(crate) fn print_generic_arg(
    db: &dyn DefDatabase,
    arg: &GenericArg,
    map: &TypesMap,
    buf: &mut dyn Write,
    edition: Edition,
) -> fmt::Result {
    match arg {
        GenericArg::Type(ty) => print_type_ref(db, *ty, map, buf, edition),
        GenericArg::Const(c) => write!(buf, "{}", c.display(db.upcast(), edition)),
        GenericArg::Lifetime(lt) => write!(buf, "{}", lt.name.display(db.upcast(), edition)),
    }
}

pub(crate) fn print_type_ref(
    db: &dyn DefDatabase,
    type_ref: TypeRefId,
    map: &TypesMap,
    buf: &mut dyn Write,
    edition: Edition,
) -> fmt::Result {
    // FIXME: deduplicate with `HirDisplay` impl
    match &map[type_ref] {
        TypeRef::Never => write!(buf, "!")?,
        TypeRef::Placeholder => write!(buf, "_")?,
        TypeRef::Tuple(fields) => {
            write!(buf, "(")?;
            for (i, field) in fields.iter().enumerate() {
                if i != 0 {
                    write!(buf, ", ")?;
                }
                print_type_ref(db, *field, map, buf, edition)?;
            }
            write!(buf, ")")?;
        }
        TypeRef::Path(path) => print_path(db, path, map, buf, edition)?,
        TypeRef::RawPtr(pointee, mtbl) => {
            let mtbl = match mtbl {
                Mutability::Shared => "*const",
                Mutability::Mut => "*mut",
            };
            write!(buf, "{mtbl} ")?;
            print_type_ref(db, *pointee, map, buf, edition)?;
        }
        TypeRef::Reference(ref_) => {
            let mtbl = match ref_.mutability {
                Mutability::Shared => "",
                Mutability::Mut => "mut ",
            };
            write!(buf, "&")?;
            if let Some(lt) = &ref_.lifetime {
                write!(buf, "{} ", lt.name.display(db.upcast(), edition))?;
            }
            write!(buf, "{mtbl}")?;
            print_type_ref(db, ref_.ty, map, buf, edition)?;
        }
        TypeRef::Array(array) => {
            write!(buf, "[")?;
            print_type_ref(db, array.ty, map, buf, edition)?;
            write!(buf, "; {}]", array.len.display(db.upcast(), edition))?;
        }
        TypeRef::Slice(elem) => {
            write!(buf, "[")?;
            print_type_ref(db, *elem, map, buf, edition)?;
            write!(buf, "]")?;
        }
        TypeRef::Fn(fn_) => {
            let ((_, return_type), args) =
                fn_.params().split_last().expect("TypeRef::Fn is missing return type");
            if fn_.is_unsafe() {
                write!(buf, "unsafe ")?;
            }
            if let Some(abi) = fn_.abi() {
                buf.write_str("extern ")?;
                buf.write_str(abi.as_str())?;
                buf.write_char(' ')?;
            }
            write!(buf, "fn(")?;
            for (i, (_, typeref)) in args.iter().enumerate() {
                if i != 0 {
                    write!(buf, ", ")?;
                }
                print_type_ref(db, *typeref, map, buf, edition)?;
            }
            if fn_.is_varargs() {
                if !args.is_empty() {
                    write!(buf, ", ")?;
                }
                write!(buf, "...")?;
            }
            write!(buf, ") -> ")?;
            print_type_ref(db, *return_type, map, buf, edition)?;
        }
        TypeRef::Macro(_ast_id) => {
            write!(buf, "<macro>")?;
        }
        TypeRef::Error => write!(buf, "{{unknown}}")?,
        TypeRef::ImplTrait(bounds) => {
            write!(buf, "impl ")?;
            print_type_bounds(db, bounds, map, buf, edition)?;
        }
        TypeRef::DynTrait(bounds) => {
            write!(buf, "dyn ")?;
            print_type_bounds(db, bounds, map, buf, edition)?;
        }
    }

    Ok(())
}

pub(crate) fn print_type_bounds(
    db: &dyn DefDatabase,
    bounds: &[TypeBound],
    map: &TypesMap,
    buf: &mut dyn Write,
    edition: Edition,
) -> fmt::Result {
    for (i, bound) in bounds.iter().enumerate() {
        if i != 0 {
            write!(buf, " + ")?;
        }

        match bound {
            TypeBound::Path(path, modifier) => {
                match modifier {
                    TraitBoundModifier::None => (),
                    TraitBoundModifier::Maybe => write!(buf, "?")?,
                }
                print_path(db, path, map, buf, edition)?;
            }
            TypeBound::ForLifetime(lifetimes, path) => {
                write!(
                    buf,
                    "for<{}> ",
                    lifetimes.iter().map(|it| it.display(db.upcast(), edition)).format(", ")
                )?;
                print_path(db, path, map, buf, edition)?;
            }
            TypeBound::Lifetime(lt) => write!(buf, "{}", lt.name.display(db.upcast(), edition))?,
            TypeBound::Use(args) => {
                write!(buf, "use<")?;
                let mut first = true;
                for arg in args {
                    if !mem::take(&mut first) {
                        write!(buf, ", ")?;
                    }
                    match arg {
                        UseArgRef::Name(it) => write!(buf, "{}", it.display(db.upcast(), edition))?,
                        UseArgRef::Lifetime(it) => {
                            write!(buf, "{}", it.name.display(db.upcast(), edition))?
                        }
                    }
                }
                write!(buf, ">")?
            }
            TypeBound::Error => write!(buf, "{{unknown}}")?,
        }
    }

    Ok(())
}
