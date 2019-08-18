use std::{fs, path::Path};

use ron;

use crate::{project_root, Mode, Result, AST, GRAMMAR};

pub fn generate(mode: Mode) -> Result<()> {
    let grammar = project_root().join(GRAMMAR);
    // let syntax_kinds = project_root().join(SYNTAX_KINDS);
    let ast = project_root().join(AST);
    generate_ast(&grammar, &ast, mode)
}

fn generate_ast(grammar_src: &Path, dst: &Path, mode: Mode) -> Result<()> {
    let src: ron::Value = {
        let text = fs::read_to_string(grammar_src)?;
        ron::de::from_str(&text)?
    };
    eprintln!("{:?}", src);
    Ok(())
}
