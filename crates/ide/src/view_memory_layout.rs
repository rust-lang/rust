use hir::{Field, HirDisplay, Layout, Semantics, Type};
use ide_db::{
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    RootDatabase,
};
use syntax::{AstNode, SyntaxKind, SyntaxToken};

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

fn get_definition(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) -> Option<Definition> {
    for token in sema.descend_into_macros(token) {
        let def = IdentClass::classify_token(sema, &token).map(IdentClass::definitions_no_ops);
        if let Some(&[x]) = def.as_deref() {
            return Some(x);
        }
    }
    None
}

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
        _ => return None,
    };

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
                    .unwrap_or_else(|| format!("{:#?}", f.name(db).as_tuple_index().unwrap())),
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

        fields.sort_by_key(|(f, _)| layout.field_offset(f.index()).unwrap_or(u64::MAX));

        if fields.len() == 0 {
            return;
        }

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

    ty.layout(db).map(|layout| {
        let item_name = match def {
            Definition::Local(l) => l.name(db).as_str().unwrap().to_owned(),
            _ => "[ROOT]".to_owned(),
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
}
