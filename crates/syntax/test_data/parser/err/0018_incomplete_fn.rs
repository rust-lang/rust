impl FnScopes {
    fn new_scope(&) -> ScopeId {
        let res = self.scopes.len();
        self.scopes.push(ScopeData { parent: None, entries: vec![] })
    }

    fn set_parent
}
