use ra_db::FileId;
use ra_syntax::ast;

use crate::db::RootDatabase;

pub fn goto_defenition(db: &RootDatabase, position: FilePosition,
) -> Cancelable<Option<Vec<NavigationTarget>>> {
    let file = db.source_file(position.file_id);
    let syntax = file.syntax();
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
        return Ok(Some(reference_defenition(db, position.file_id, name_ref)));
    }
    if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset)  {
        return Ok(Some(name_defenition(db, position.file_idname)));
    }
    Ok(None)
}

fn reference_defenition(db: &RootDatabase, file_id: FileId, name_ref: ast::NameRef) -> Cancelable<Vec<Nav>> {
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
        let mut rr = ReferenceResolution::new(name_ref.syntax().range());
        if let Some(fn_descr) =
            source_binder::function_from_child_node(self, position.file_id, name_ref.syntax())?
        {
            let scope = fn_descr.scopes(self);
            // First try to resolve the symbol locally
            if let Some(entry) = scope.resolve_local_name(name_ref) {
                rr.resolves_to.push(NavigationTarget {
                    file_id: position.file_id,
                    name: entry.name().to_string().into(),
                    range: entry.ptr().range(),
                    kind: NAME,
                    ptr: None,
                });
                return Ok(Some(rr));
            };
        }
        // If that fails try the index based approach.
        rr.resolves_to.extend(
            self.index_resolve(name_ref)?
                .into_iter()
                .map(NavigationTarget::from_symbol),
        );
        return Ok(Some(rr));
    }
        if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
            let mut rr = ReferenceResolution::new(name.syntax().range());
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    if let Some(child_module) =
                        source_binder::module_from_declaration(self, position.file_id, module)?
                    {
                        let file_id = child_module.file_id();
                        let name = match child_module.name() {
                            Some(name) => name.to_string().into(),
                            None => "".into(),
                        };
                        let symbol = NavigationTarget {
                            file_id,
                            name,
                            range: TextRange::offset_len(0.into(), 0.into()),
                            kind: MODULE,
                            ptr: None,
                        };
                        rr.resolves_to.push(symbol);
                        return Ok(Some(rr));
                    }
                }
            }
        }
        Ok(None)

}
