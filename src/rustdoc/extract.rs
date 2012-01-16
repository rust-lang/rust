#[doc = "Converting the Rust AST to the rustdoc document model"];

import rustc::syntax::ast;

#[doc = "Converts the Rust AST to the rustdoc document model"]
fn extract(crate: @ast::crate) -> doc::cratedoc {
    {
        mods: []
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn extract_empty_crate() {
        let source = ""; // empty crate
        let ast = parse::from_str(source);
        let doc = extract(ast);
        assert doc.mods == [];
    }
}