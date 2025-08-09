use ide_db::{famous_defs::FamousDefs, helpers::mod_path_to_ast, traits::resolve_target_trait};
use syntax::ast::{self, AstNode, HasGenericArgs, HasName};

use crate::{AssistContext, AssistId, Assists};

// FIXME: this should be a diagnostic

// Assist: convert_into_to_from
//
// Converts an Into impl to an equivalent From impl.
//
// ```
// # //- minicore: from
// impl $0Into<Thing> for usize {
//     fn into(self) -> Thing {
//         Thing {
//             b: self.to_string(),
//             a: self
//         }
//     }
// }
// ```
// ->
// ```
// impl From<usize> for Thing {
//     fn from(val: usize) -> Self {
//         Thing {
//             b: val.to_string(),
//             a: val
//         }
//     }
// }
// ```
pub(crate) fn convert_into_to_from(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let impl_ = ctx.find_node_at_offset::<ast::Impl>()?;
    let src_type = impl_.self_ty()?;
    let ast_trait = impl_.trait_()?;

    let module = ctx.sema.scope(impl_.syntax())?.module();

    let trait_ = resolve_target_trait(&ctx.sema, &impl_)?;
    if trait_ != FamousDefs(&ctx.sema, module.krate()).core_convert_Into()? {
        return None;
    }

    let cfg = ctx.config.import_path_config();

    let src_type_path = {
        let src_type_path = src_type.syntax().descendants().find_map(ast::Path::cast)?;
        let src_type_def = match ctx.sema.resolve_path(&src_type_path) {
            Some(hir::PathResolution::Def(module_def)) => module_def,
            _ => return None,
        };
        mod_path_to_ast(
            &module.find_path(ctx.db(), src_type_def, cfg)?,
            module.krate().edition(ctx.db()),
        )
    };

    let dest_type = match &ast_trait {
        ast::Type::PathType(path) => {
            path.path()?.segment()?.generic_arg_list()?.generic_args().next()?
        }
        _ => return None,
    };

    let into_fn = impl_.assoc_item_list()?.assoc_items().find_map(|item| {
        if let ast::AssocItem::Fn(f) = item
            && f.name()?.text() == "into"
        {
            return Some(f);
        };
        None
    })?;

    let into_fn_name = into_fn.name()?;
    let into_fn_params = into_fn.param_list()?;
    let into_fn_return = into_fn.ret_type()?;

    let selfs = into_fn
        .body()?
        .syntax()
        .descendants()
        .filter_map(ast::NameRef::cast)
        .filter(|name| name.text() == "self" || name.text() == "Self");

    acc.add(
        AssistId::refactor_rewrite("convert_into_to_from"),
        "Convert Into to From",
        impl_.syntax().text_range(),
        |builder| {
            builder.replace(src_type.syntax().text_range(), dest_type.to_string());
            builder.replace(ast_trait.syntax().text_range(), format!("From<{src_type}>"));
            builder.replace(into_fn_return.syntax().text_range(), "-> Self");
            builder.replace(into_fn_params.syntax().text_range(), format!("(val: {src_type})"));
            builder.replace(into_fn_name.syntax().text_range(), "from");

            for s in selfs {
                match s.text().as_ref() {
                    "self" => builder.replace(s.syntax().text_range(), "val"),
                    "Self" => builder.replace(s.syntax().text_range(), src_type_path.to_string()),
                    _ => {}
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn convert_into_to_from_converts_a_struct() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
struct Thing {
    a: String,
    b: usize
}

impl $0core::convert::Into<Thing> for usize {
    fn into(self) -> Thing {
        Thing {
            b: self.to_string(),
            a: self
        }
    }
}
"#,
            r#"
struct Thing {
    a: String,
    b: usize
}

impl From<usize> for Thing {
    fn from(val: usize) -> Self {
        Thing {
            b: val.to_string(),
            a: val
        }
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_converts_enums() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
enum Thing {
    Foo(String),
    Bar(String)
}

impl $0core::convert::Into<String> for Thing {
    fn into(self) -> String {
        match self {
            Self::Foo(s) => s,
            Self::Bar(s) => s
        }
    }
}
"#,
            r#"
enum Thing {
    Foo(String),
    Bar(String)
}

impl From<Thing> for String {
    fn from(val: Thing) -> Self {
        match val {
            Thing::Foo(s) => s,
            Thing::Bar(s) => s
        }
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_on_enum_with_lifetimes() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
enum Thing<'a> {
    Foo(&'a str),
    Bar(&'a str)
}

impl<'a> $0core::convert::Into<&'a str> for Thing<'a> {
    fn into(self) -> &'a str {
        match self {
            Self::Foo(s) => s,
            Self::Bar(s) => s
        }
    }
}
"#,
            r#"
enum Thing<'a> {
    Foo(&'a str),
    Bar(&'a str)
}

impl<'a> From<Thing<'a>> for &'a str {
    fn from(val: Thing<'a>) -> Self {
        match val {
            Thing::Foo(s) => s,
            Thing::Bar(s) => s
        }
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_works_on_references() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
struct Thing(String);

impl $0core::convert::Into<String> for &Thing {
    fn into(self) -> Thing {
        self.0.clone()
    }
}
"#,
            r#"
struct Thing(String);

impl From<&Thing> for String {
    fn from(val: &Thing) -> Self {
        val.0.clone()
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_works_on_qualified_structs() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
mod things {
    pub struct Thing(String);
    pub struct BetterThing(String);
}

impl $0core::convert::Into<things::BetterThing> for &things::Thing {
    fn into(self) -> Thing {
        things::BetterThing(self.0.clone())
    }
}
"#,
            r#"
mod things {
    pub struct Thing(String);
    pub struct BetterThing(String);
}

impl From<&things::Thing> for things::BetterThing {
    fn from(val: &things::Thing) -> Self {
        things::BetterThing(val.0.clone())
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_works_on_qualified_enums() {
        check_assist(
            convert_into_to_from,
            r#"
//- minicore: from
mod things {
    pub enum Thing {
        A(String)
    }
    pub struct BetterThing {
        B(String)
    }
}

impl $0core::convert::Into<things::BetterThing> for &things::Thing {
    fn into(self) -> Thing {
        match self {
            Self::A(s) => things::BetterThing::B(s)
        }
    }
}
"#,
            r#"
mod things {
    pub enum Thing {
        A(String)
    }
    pub struct BetterThing {
        B(String)
    }
}

impl From<&things::Thing> for things::BetterThing {
    fn from(val: &things::Thing) -> Self {
        match val {
            things::Thing::A(s) => things::BetterThing::B(s)
        }
    }
}
"#,
        )
    }

    #[test]
    fn convert_into_to_from_not_applicable_on_any_trait_named_into() {
        check_assist_not_applicable(
            convert_into_to_from,
            r#"
//- minicore: from
pub trait Into<T> {
    pub fn into(self) -> T;
}

struct Thing {
    a: String,
}

impl $0Into<Thing> for String {
    fn into(self) -> Thing {
        Thing {
            a: self
        }
    }
}
"#,
        );
    }
}
