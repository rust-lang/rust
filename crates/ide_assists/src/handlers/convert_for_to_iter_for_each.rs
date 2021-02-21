use ast::LoopBodyOwner;
use ide_db::helpers::FamousDefs;
use stdx::format_to;
use syntax::{ast, AstNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_for_to_iter_for_each
//
// Converts a for loop into a for_each loop on the Iterator.
//
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     for $0v in x {
//         let y = v * 2;
//     }
// }
// ```
// ->
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     x.into_iter().for_each(|v| {
//         let y = v * 2;
//     });
// }
// ```
pub(crate) fn convert_for_to_iter_for_each(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let for_loop = ctx.find_node_at_offset::<ast::ForExpr>()?;
    let iterable = for_loop.iterable()?;
    let pat = for_loop.pat()?;
    let body = for_loop.loop_body()?;

    let mut buf = String::new();

    if impls_core_iter(&ctx.sema, &iterable) {
        buf += &iterable.to_string();
    } else {
        match iterable {
            ast::Expr::RefExpr(r) => {
                if r.mut_token().is_some() {
                    format_to!(buf, "{}.iter_mut()", r.expr()?);
                } else {
                    format_to!(buf, "{}.iter()", r.expr()?);
                }
            }
            _ => format_to!(buf, "{}.into_iter()", iterable),
        }
    }

    format_to!(buf, ".for_each(|{}| {});", pat, body);

    acc.add(
        AssistId("convert_for_to_iter_for_each", AssistKind::RefactorRewrite),
        "Convert a for loop into an Iterator::for_each",
        for_loop.syntax().text_range(),
        |builder| builder.replace(for_loop.syntax().text_range(), buf),
    )
}

fn impls_core_iter(sema: &hir::Semantics<ide_db::RootDatabase>, iterable: &ast::Expr) -> bool {
    let it_typ = if let Some(i) = sema.type_of_expr(iterable) {
        i
    } else {
        return false;
    };
    let module = if let Some(m) = sema.scope(iterable.syntax()).module() {
        m
    } else {
        return false;
    };
    let krate = module.krate();
    if let Some(iter_trait) = FamousDefs(sema, Some(krate)).core_iter_Iterator() {
        return it_typ.impls_trait(sema.db, iter_trait, &[]);
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    const EMPTY_ITER_FIXTURE: &'static str = r"
//- /lib.rs deps:core crate:empty_iter
pub struct EmptyIter;
impl Iterator for EmptyIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> { None }
}

pub struct Empty;
impl Empty {
    pub fn iter(&self) -> EmptyIter { EmptyIter }
}
";

    #[test]
    fn test_not_for() {
        check_assist_not_applicable(
            convert_for_to_iter_for_each,
            r"
let mut x = vec![1, 2, 3];
x.iter_mut().$0for_each(|v| *v *= 2);
        ",
        )
    }

    #[test]
    fn test_simple_for() {
        check_assist(
            convert_for_to_iter_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    for $0v in x {
        v *= 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    x.into_iter().for_each(|v| {
        v *= 2;
    });
}",
        )
    }

    #[test]
    fn test_for_borrowed() {
        check_assist(
            convert_for_to_iter_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    for $0v in &x {
        let a = v * 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    x.iter().for_each(|v| {
        let a = v * 2;
    });
}",
        )
    }

    #[test]
    fn test_for_borrowed_mut() {
        check_assist(
            convert_for_to_iter_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    for $0v in &mut x {
        *v *= 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    x.iter_mut().for_each(|v| {
        *v *= 2;
    });
}",
        )
    }

    #[test]
    fn test_for_borrowed_mut_behind_var() {
        check_assist(
            convert_for_to_iter_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    let y = &mut x;
    for $0v in y {
        *v *= 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    let y = &mut x;
    y.into_iter().for_each(|v| {
        *v *= 2;
    });
}",
        )
    }

    #[test]
    fn test_take() {
        let before = r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    for$0 a in x.iter().take(1) {
        println!("{}", a);
    }
}
"#;
        let after = r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().take(1).for_each(|a| {
        println!("{}", a);
    });
}
"#;
        let before = &format!(
            "//- /main.rs crate:main deps:core,empty_iter{}{}{}",
            before,
            FamousDefs::FIXTURE,
            EMPTY_ITER_FIXTURE
        );
        check_assist(convert_for_to_iter_for_each, before, after);
    }
}
