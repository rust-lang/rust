//! Display and pretty printing routines.

use std::fmt::{self, Write};

use hir_expand::{db::ExpandDatabase, mod_path::PathKind};
use intern::Interned;
use itertools::Itertools;

use crate::{
    path::{GenericArg, GenericArgs, Path},
    type_ref::{Mutability, TraitBoundModifier, TypeBound, TypeRef},
};

pub(crate) fn print_path(db: &dyn ExpandDatabase, path: &Path, buf: &mut dyn Write) -> fmt::Result {
    if let Path::LangItem(x) = path {
        return write!(buf, "$lang_item::{x:?}");
    }
    match path.type_anchor() {
        Some(anchor) => {
            write!(buf, "<")?;
            print_type_ref(db, anchor, buf)?;
            write!(buf, ">::")?;
        }
        None => match path.kind() {
            PathKind::Plain => {}
            PathKind::Super(0) => write!(buf, "self")?,
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

        write!(buf, "{}", segment.name.display(db))?;
        if let Some(generics) = segment.args_and_bindings {
            write!(buf, "::<")?;
            print_generic_args(db, generics, buf)?;

            write!(buf, ">")?;
        }
    }

    Ok(())
}

pub(crate) fn print_generic_args(
    db: &dyn ExpandDatabase,
    generics: &GenericArgs,
    buf: &mut dyn Write,
) -> fmt::Result {
    let mut first = true;
    let args = if generics.has_self_type {
        let (self_ty, args) = generics.args.split_first().unwrap();
        write!(buf, "Self=")?;
        print_generic_arg(db, self_ty, buf)?;
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
        print_generic_arg(db, arg, buf)?;
    }
    for binding in generics.bindings.iter() {
        if !first {
            write!(buf, ", ")?;
        }
        first = false;
        write!(buf, "{}", binding.name.display(db))?;
        if !binding.bounds.is_empty() {
            write!(buf, ": ")?;
            print_type_bounds(db, &binding.bounds, buf)?;
        }
        if let Some(ty) = &binding.type_ref {
            write!(buf, " = ")?;
            print_type_ref(db, ty, buf)?;
        }
    }
    Ok(())
}

pub(crate) fn print_generic_arg(
    db: &dyn ExpandDatabase,
    arg: &GenericArg,
    buf: &mut dyn Write,
) -> fmt::Result {
    match arg {
        GenericArg::Type(ty) => print_type_ref(db, ty, buf),
        GenericArg::Const(c) => write!(buf, "{}", c.display(db)),
        GenericArg::Lifetime(lt) => write!(buf, "{}", lt.name.display(db)),
    }
}

pub(crate) fn print_type_ref(
    db: &dyn ExpandDatabase,
    type_ref: &TypeRef,
    buf: &mut dyn Write,
) -> fmt::Result {
    // FIXME: deduplicate with `HirDisplay` impl
    match type_ref {
        TypeRef::Never => write!(buf, "!")?,
        TypeRef::Placeholder => write!(buf, "_")?,
        TypeRef::Tuple(fields) => {
            write!(buf, "(")?;
            for (i, field) in fields.iter().enumerate() {
                if i != 0 {
                    write!(buf, ", ")?;
                }
                print_type_ref(db, field, buf)?;
            }
            write!(buf, ")")?;
        }
        TypeRef::Path(path) => print_path(db, path, buf)?,
        TypeRef::RawPtr(pointee, mtbl) => {
            let mtbl = match mtbl {
                Mutability::Shared => "*const",
                Mutability::Mut => "*mut",
            };
            write!(buf, "{mtbl} ")?;
            print_type_ref(db, pointee, buf)?;
        }
        TypeRef::Reference(pointee, lt, mtbl) => {
            let mtbl = match mtbl {
                Mutability::Shared => "",
                Mutability::Mut => "mut ",
            };
            write!(buf, "&")?;
            if let Some(lt) = lt {
                write!(buf, "{} ", lt.name.display(db))?;
            }
            write!(buf, "{mtbl}")?;
            print_type_ref(db, pointee, buf)?;
        }
        TypeRef::Array(elem, len) => {
            write!(buf, "[")?;
            print_type_ref(db, elem, buf)?;
            write!(buf, "; {}]", len.display(db))?;
        }
        TypeRef::Slice(elem) => {
            write!(buf, "[")?;
            print_type_ref(db, elem, buf)?;
            write!(buf, "]")?;
        }
        TypeRef::Fn(args_and_ret, varargs, is_unsafe) => {
            let ((_, return_type), args) =
                args_and_ret.split_last().expect("TypeRef::Fn is missing return type");
            if *is_unsafe {
                write!(buf, "unsafe ")?;
            }
            write!(buf, "fn(")?;
            for (i, (_, typeref)) in args.iter().enumerate() {
                if i != 0 {
                    write!(buf, ", ")?;
                }
                print_type_ref(db, typeref, buf)?;
            }
            if *varargs {
                if !args.is_empty() {
                    write!(buf, ", ")?;
                }
                write!(buf, "...")?;
            }
            write!(buf, ") -> ")?;
            print_type_ref(db, return_type, buf)?;
        }
        TypeRef::Macro(_ast_id) => {
            write!(buf, "<macro>")?;
        }
        TypeRef::Error => write!(buf, "{{unknown}}")?,
        TypeRef::ImplTrait(bounds) => {
            write!(buf, "impl ")?;
            print_type_bounds(db, bounds, buf)?;
        }
        TypeRef::DynTrait(bounds) => {
            write!(buf, "dyn ")?;
            print_type_bounds(db, bounds, buf)?;
        }
    }

    Ok(())
}

pub(crate) fn print_type_bounds(
    db: &dyn ExpandDatabase,
    bounds: &[Interned<TypeBound>],
    buf: &mut dyn Write,
) -> fmt::Result {
    for (i, bound) in bounds.iter().enumerate() {
        if i != 0 {
            write!(buf, " + ")?;
        }

        match bound.as_ref() {
            TypeBound::Path(path, modifier) => {
                match modifier {
                    TraitBoundModifier::None => (),
                    TraitBoundModifier::Maybe => write!(buf, "?")?,
                }
                print_path(db, path, buf)?;
            }
            TypeBound::ForLifetime(lifetimes, path) => {
                write!(buf, "for<{}> ", lifetimes.iter().map(|it| it.display(db)).format(", "))?;
                print_path(db, path, buf)?;
            }
            TypeBound::Lifetime(lt) => write!(buf, "{}", lt.name.display(db))?,
            TypeBound::Error => write!(buf, "{{unknown}}")?,
        }
    }

    Ok(())
}
