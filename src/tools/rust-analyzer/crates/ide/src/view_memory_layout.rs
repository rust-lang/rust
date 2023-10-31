use std::fmt;

use hir::{Field, HirDisplay, Layout, Semantics, Type};
use ide_db::{
    defs::Definition,
    helpers::{get_definition, pick_best_token},
    RootDatabase,
};
use syntax::{AstNode, SyntaxKind};

use crate::FilePosition;

pub struct MemoryLayoutNode {
    pub item_name: String,
    pub typename: String,
    pub size: u64,
    pub alignment: u64,
    pub offset: u64,
    pub parent_idx: i64,
    pub children_start: i64,
    pub children_len: u64,
}

pub struct RecursiveMemoryLayout {
    pub nodes: Vec<MemoryLayoutNode>,
}

// NOTE: this is currently strictly for testing and so isn't super useful as a visualization tool, however it could be adapted to become one?
impl fmt::Display for RecursiveMemoryLayout {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn process(
            fmt: &mut fmt::Formatter<'_>,
            nodes: &Vec<MemoryLayoutNode>,
            idx: usize,
            depth: usize,
        ) -> fmt::Result {
            let mut out = "\t".repeat(depth);
            let node = &nodes[idx];
            out += &format!(
                "{}: {} (size: {}, align: {}, field offset: {})\n",
                node.item_name, node.typename, node.size, node.alignment, node.offset
            );
            write!(fmt, "{}", out)?;
            if node.children_start != -1 {
                for j in nodes[idx].children_start
                    ..(nodes[idx].children_start + nodes[idx].children_len as i64)
                {
                    process(fmt, nodes, j as usize, depth + 1)?;
                }
            }
            Ok(())
        }

        process(fmt, &self.nodes, 0, 0)
    }
}

enum FieldOrTupleIdx {
    Field(Field),
    TupleIdx(usize),
}

impl FieldOrTupleIdx {
    fn name(&self, db: &RootDatabase) -> String {
        match *self {
            FieldOrTupleIdx::Field(f) => f
                .name(db)
                .as_str()
                .map(|s| s.to_owned())
                .unwrap_or_else(|| format!(".{}", f.name(db).as_tuple_index().unwrap())),
            FieldOrTupleIdx::TupleIdx(i) => format!(".{i}").to_owned(),
        }
    }

    fn index(&self) -> usize {
        match *self {
            FieldOrTupleIdx::Field(f) => f.index(),
            FieldOrTupleIdx::TupleIdx(i) => i,
        }
    }
}

// Feature: View Memory Layout
//
// Displays the recursive memory layout of a datatype.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: View Memory Layout**
// |===
pub(crate) fn view_memory_layout(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RecursiveMemoryLayout> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id);
    let token =
        pick_best_token(file.syntax().token_at_offset(position.offset), |kind| match kind {
            SyntaxKind::IDENT => 3,
            _ => 0,
        })?;

    let def = get_definition(&sema, token)?;

    let ty = match def {
        Definition::Adt(it) => it.ty(db),
        Definition::TypeAlias(it) => it.ty(db),
        Definition::BuiltinType(it) => it.ty(db),
        Definition::SelfType(it) => it.self_ty(db),
        Definition::Local(it) => it.ty(db),
        Definition::Field(it) => it.ty(db),
        Definition::Const(it) => it.ty(db),
        Definition::Static(it) => it.ty(db),
        _ => return None,
    };

    fn read_layout(
        nodes: &mut Vec<MemoryLayoutNode>,
        db: &RootDatabase,
        ty: &Type,
        layout: &Layout,
        parent_idx: usize,
    ) {
        let mut fields = ty
            .fields(db)
            .into_iter()
            .map(|(f, ty)| (FieldOrTupleIdx::Field(f), ty))
            .chain(
                ty.tuple_fields(db)
                    .into_iter()
                    .enumerate()
                    .map(|(i, ty)| (FieldOrTupleIdx::TupleIdx(i), ty)),
            )
            .collect::<Vec<_>>();

        if fields.len() == 0 {
            return;
        }

        fields.sort_by_key(|(f, _)| layout.field_offset(f.index()).unwrap());

        let children_start = nodes.len();
        nodes[parent_idx].children_start = children_start as i64;
        nodes[parent_idx].children_len = fields.len() as u64;

        for (field, child_ty) in fields.iter() {
            if let Ok(child_layout) = child_ty.layout(db) {
                nodes.push(MemoryLayoutNode {
                    item_name: field.name(db),
                    typename: child_ty.display(db).to_string(),
                    size: child_layout.size(),
                    alignment: child_layout.align(),
                    offset: layout.field_offset(field.index()).unwrap_or(0),
                    parent_idx: parent_idx as i64,
                    children_start: -1,
                    children_len: 0,
                });
            } else {
                nodes.push(MemoryLayoutNode {
                    item_name: field.name(db)
                        + format!("(no layout data: {:?})", child_ty.layout(db).unwrap_err())
                            .as_ref(),
                    typename: child_ty.display(db).to_string(),
                    size: 0,
                    offset: 0,
                    alignment: 0,
                    parent_idx: parent_idx as i64,
                    children_start: -1,
                    children_len: 0,
                });
            }
        }

        for (i, (_, child_ty)) in fields.iter().enumerate() {
            if let Ok(child_layout) = child_ty.layout(db) {
                read_layout(nodes, db, &child_ty, &child_layout, children_start + i);
            }
        }
    }

    ty.layout(db)
        .map(|layout| {
            let item_name = match def {
                // def is a datatype
                Definition::Adt(_)
                | Definition::TypeAlias(_)
                | Definition::BuiltinType(_)
                | Definition::SelfType(_) => "[ROOT]".to_owned(),

                // def is an item
                def => def
                    .name(db)
                    .map(|n| {
                        n.as_str()
                            .map(|s| s.to_owned())
                            .unwrap_or_else(|| format!(".{}", n.as_tuple_index().unwrap()))
                    })
                    .unwrap_or("[ROOT]".to_owned()),
            };

            let typename = ty.display(db).to_string();

            let mut nodes = vec![MemoryLayoutNode {
                item_name,
                typename: typename.clone(),
                size: layout.size(),
                offset: 0,
                alignment: layout.align(),
                parent_idx: -1,
                children_start: -1,
                children_len: 0,
            }];
            read_layout(&mut nodes, db, &ty, &layout, 0);

            RecursiveMemoryLayout { nodes }
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fixture;
    use expect_test::expect;

    fn make_memory_layout(ra_fixture: &str) -> Option<RecursiveMemoryLayout> {
        let (analysis, position, _) = fixture::annotations(ra_fixture);

        view_memory_layout(&analysis.db, position)
    }

    #[test]
    fn view_memory_layout_none() {
        assert!(make_memory_layout(r#"$0"#).is_none());
        assert!(make_memory_layout(r#"stru$0ct Blah {}"#).is_none());
    }

    #[test]
    fn view_memory_layout_primitive() {
        expect![[r#"
            foo: i32 (size: 4, align: 4, field offset: 0)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
fn main() {
    let foo$0 = 109; // default i32
}
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_constant() {
        expect![[r#"
            BLAH: bool (size: 1, align: 1, field offset: 0)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
const BLAH$0: bool = 0;
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_static() {
        expect![[r#"
            BLAH: bool (size: 1, align: 1, field offset: 0)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
static BLAH$0: bool = 0;
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_tuple() {
        expect![[r#"
            x: (f64, u8, i64) (size: 24, align: 8, field offset: 0)
            	.0: f64 (size: 8, align: 8, field offset: 0)
            	.1: u8 (size: 1, align: 1, field offset: 8)
            	.2: i64 (size: 8, align: 8, field offset: 16)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
fn main() {
    let x$0 = (101.0, 111u8, 119i64);
}
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_c_struct() {
        expect![[r#"
            [ROOT]: Blah (size: 16, align: 4, field offset: 0)
            	a: u32 (size: 4, align: 4, field offset: 0)
            	b: (i32, u8) (size: 8, align: 4, field offset: 4)
            		.0: i32 (size: 4, align: 4, field offset: 0)
            		.1: u8 (size: 1, align: 1, field offset: 4)
            	c: i8 (size: 1, align: 1, field offset: 12)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
#[repr(C)]
struct Blah$0 {
    a: u32,
    b: (i32, u8),
    c: i8,
}
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_struct() {
        expect![[r#"
            [ROOT]: Blah (size: 16, align: 4, field offset: 0)
            	b: (i32, u8) (size: 8, align: 4, field offset: 0)
            		.0: i32 (size: 4, align: 4, field offset: 0)
            		.1: u8 (size: 1, align: 1, field offset: 4)
            	a: u32 (size: 4, align: 4, field offset: 8)
            	c: i8 (size: 1, align: 1, field offset: 12)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
struct Blah$0 {
    a: u32,
    b: (i32, u8),
    c: i8,
}
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_member() {
        expect![[r#"
            a: bool (size: 1, align: 1, field offset: 0)
        "#]]
        .assert_eq(
            &make_memory_layout(
                r#"
#[repr(C)]
struct Oof {
    a$0: bool,
}
"#,
            )
            .unwrap()
            .to_string(),
        );
    }

    #[test]
    fn view_memory_layout_alias() {
        let ml_a = make_memory_layout(
            r#"
struct X {
    a: u32,
    b: i8,
    c: (f32, f32),
}

type Foo$0 = X;
"#,
        )
        .unwrap();

        let ml_b = make_memory_layout(
            r#"
struct X$0 {
    a: u32,
    b: i8,
    c: (f32, f32),
}
"#,
        )
        .unwrap();

        assert_eq!(ml_a.to_string(), ml_b.to_string());
    }
}
