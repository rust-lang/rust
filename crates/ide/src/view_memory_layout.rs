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

    fn make_memory_layout(ra_fixture: &str) -> Option<RecursiveMemoryLayout> {
        let (analysis, position, _) = fixture::annotations(ra_fixture);

        view_memory_layout(&analysis.db, position)
    }

    fn check_item_info<T>(node: &MemoryLayoutNode, item_name: &str, check_typename: bool) {
        assert_eq!(node.item_name, item_name);
        assert_eq!(node.size, core::mem::size_of::<T>() as u64);
        assert_eq!(node.alignment, core::mem::align_of::<T>() as u64);
        if check_typename {
            assert_eq!(node.typename, std::any::type_name::<T>());
        }
    }

    #[test]
    fn view_memory_layout_none() {
        assert!(make_memory_layout(r#"$0"#).is_none());
        assert!(make_memory_layout(r#"stru$0ct Blah {}"#).is_none());
    }

    #[test]
    fn view_memory_layout_primitive() {
        let ml = make_memory_layout(
            r#"
fn main() {
    let foo$0 = 109; // default i32
}
"#,
        )
        .unwrap();

        assert_eq!(ml.nodes.len(), 1);
        assert_eq!(ml.nodes[0].parent_idx, -1);
        assert_eq!(ml.nodes[0].children_start, -1);
        check_item_info::<i32>(&ml.nodes[0], "foo", true);
        assert_eq!(ml.nodes[0].offset, 0);
    }

    #[test]
    fn view_memory_layout_constant() {
        let ml = make_memory_layout(
            r#"
const BLAH$0: bool = 0;
"#,
        )
        .unwrap();

        assert_eq!(ml.nodes.len(), 1);
        assert_eq!(ml.nodes[0].parent_idx, -1);
        assert_eq!(ml.nodes[0].children_start, -1);
        check_item_info::<bool>(&ml.nodes[0], "BLAH", true);
        assert_eq!(ml.nodes[0].offset, 0);
    }

    #[test]
    fn view_memory_layout_static() {
        let ml = make_memory_layout(
            r#"
static BLAH$0: bool = 0;
"#,
        )
        .unwrap();

        assert_eq!(ml.nodes.len(), 1);
        assert_eq!(ml.nodes[0].parent_idx, -1);
        assert_eq!(ml.nodes[0].children_start, -1);
        check_item_info::<bool>(&ml.nodes[0], "BLAH", true);
        assert_eq!(ml.nodes[0].offset, 0);
    }

    #[test]
    fn view_memory_layout_tuple() {
        let ml = make_memory_layout(
            r#"
fn main() {
    let x$0 = (101.0, 111u8, 119i64);
}
        "#,
        )
        .unwrap();

        assert_eq!(ml.nodes.len(), 4);
        assert_eq!(ml.nodes[0].children_start, 1);
        assert_eq!(ml.nodes[0].children_len, 3);
        check_item_info::<(f64, u8, i64)>(&ml.nodes[0], "x", true);
    }

    #[test]
    fn view_memory_layout_struct() {
        let ml = make_memory_layout(
            r#"
#[repr(C)]
struct Blah$0 {
    a: u32,
    b: (i32, u8),
    c: i8,
}
"#,
        )
        .unwrap();

        #[repr(C)] // repr C makes this testable, rustc doesn't enforce a layout otherwise ;-;
        struct Blah {
            a: u32,
            b: (i32, u8),
            c: i8,
        }

        assert_eq!(ml.nodes.len(), 6);
        check_item_info::<Blah>(&ml.nodes[0], "[ROOT]", false);
        assert_eq!(ml.nodes[0].offset, 0);

        check_item_info::<u32>(&ml.nodes[1], "a", true);
        assert_eq!(ml.nodes[1].offset, 0);

        check_item_info::<(i32, u8)>(&ml.nodes[2], "b", true);
        assert_eq!(ml.nodes[2].offset, 4);

        check_item_info::<i8>(&ml.nodes[3], "c", true);
        assert_eq!(ml.nodes[3].offset, 12);
    }

    #[test]
    fn view_memory_layout_member() {
        let ml = make_memory_layout(
            r#"
struct Oof {
    a$0: bool
}
"#,
        )
        .unwrap();

        assert_eq!(ml.nodes.len(), 1);
        assert_eq!(ml.nodes[0].parent_idx, -1);
        assert_eq!(ml.nodes[0].children_start, -1);
        check_item_info::<bool>(&ml.nodes[0], "a", true);
        // NOTE: this should not give the memory layout relative to the parent structure, but the type referred to by the member variable alone.
        assert_eq!(ml.nodes[0].offset, 0);
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

        ml_a.nodes.iter().zip(ml_b.nodes.iter()).for_each(|(a, b)| {
            assert_eq!(a.item_name, b.item_name);
            assert_eq!(a.typename, b.typename);
            assert_eq!(a.size, b.size);
            assert_eq!(a.alignment, b.alignment);
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.parent_idx, b.parent_idx);
            assert_eq!(a.children_start, b.children_start);
            assert_eq!(a.children_len, b.children_len);
        })
    }
}
