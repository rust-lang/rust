//! Structural editing for ast using `SyntaxEditor`

use crate::{
    ast::make, ast::AstNode, ast::Fn, ast::GenericParam, ast::HasGenericParams, ast::HasName,
    syntax_editor::Position, syntax_editor::SyntaxEditor, SyntaxKind,
};

impl SyntaxEditor {
    /// Adds a new generic param to the function using `SyntaxEditor`
    pub fn syntax_editor_add_generic_param(&mut self, function: Fn, new_param: GenericParam) {
        match function.generic_param_list() {
            Some(generic_param_list) => match generic_param_list.generic_params().last() {
                Some(last_param) => {
                    // There exists a generic param list and it's not empty
                    let position = generic_param_list.r_angle_token().map_or_else(
                        || Position::last_child_of(function.syntax()),
                        Position::before,
                    );

                    if last_param
                        .syntax()
                        .next_sibling_or_token()
                        .map_or(false, |it| it.kind() == SyntaxKind::COMMA)
                    {
                        self.insert(
                            Position::after(last_param.syntax()),
                            new_param.syntax().clone(),
                        );
                        self.insert(
                            Position::after(last_param.syntax()),
                            make::token(SyntaxKind::WHITESPACE),
                        );
                        self.insert(
                            Position::after(last_param.syntax()),
                            make::token(SyntaxKind::COMMA),
                        );
                    } else {
                        let elements = vec![
                            make::token(SyntaxKind::COMMA).into(),
                            make::token(SyntaxKind::WHITESPACE).into(),
                            new_param.syntax().clone().into(),
                        ];
                        self.insert_all(position, elements);
                    }
                }
                None => {
                    // There exists a generic param list but it's empty
                    let position = Position::after(generic_param_list.l_angle_token().unwrap());
                    self.insert(position, new_param.syntax());
                }
            },
            None => {
                // There was no generic param list
                let position = if let Some(name) = function.name() {
                    Position::after(name.syntax)
                } else if let Some(fn_token) = function.fn_token() {
                    Position::after(fn_token)
                } else if let Some(param_list) = function.param_list() {
                    Position::before(param_list.syntax)
                } else {
                    Position::last_child_of(function.syntax())
                };
                let elements = vec![
                    make::token(SyntaxKind::L_ANGLE).into(),
                    new_param.syntax().clone().into(),
                    make::token(SyntaxKind::R_ANGLE).into(),
                ];
                self.insert_all(position, elements);
            }
        }
    }
}
