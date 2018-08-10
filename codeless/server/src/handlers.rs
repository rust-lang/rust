use libanalysis::World;
use libeditor;
use {req, Result};

pub fn handle_syntax_tree(
    world: World,
    params: req::SyntaxTreeParams
) -> Result<String> {
    let path = params.text_document.uri.to_file_path()
        .map_err(|()| format_err!("invalid path"))?;
    let file = world.file_syntax(&path)?;
    Ok(libeditor::syntax_tree(&file))
}
