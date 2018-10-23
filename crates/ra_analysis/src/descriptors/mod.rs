pub(crate) mod module;

use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    text_utils::is_subrange,
};

#[derive(Debug, Clone)]
pub struct FnDescriptor {
    pub name: String,
    pub label: String,
    pub ret_type: Option<String>,
    pub params: Vec<String>,
}

impl FnDescriptor {
    pub fn new(node: ast::FnDef) -> Option<Self> {
        let name = node.name()?.text().to_string();

        // Strip the body out for the label.
        let label: String = if let Some(body) = node.body() {
            let body_range = body.syntax().range();
            let label: String = node
                .syntax()
                .children()
                .filter(|child| !is_subrange(body_range, child.range()))
                .map(|node| node.text().to_string())
                .collect();
            label
        } else {
            node.syntax().text().to_string()
        };

        let params = FnDescriptor::param_list(node);
        let ret_type = node.ret_type().map(|r| r.syntax().text().to_string());

        Some(FnDescriptor {
            name,
            ret_type,
            params,
            label,
        })
    }

    fn param_list(node: ast::FnDef) -> Vec<String> {
        let mut res = vec![];
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                res.push(self_param.syntax().text().to_string())
            }

            // Maybe use param.pat here? See if we can just extract the name?
            //res.extend(param_list.params().map(|p| p.syntax().text().to_string()));
            res.extend(
                param_list
                    .params()
                    .filter_map(|p| p.pat())
                    .map(|pat| pat.syntax().text().to_string()),
            );
        }
        res
    }
}
