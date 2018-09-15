use libsyntax2::{
    SmolStr,
    ast::{self, NameOwner},
};

#[derive(Debug, Hash)]
pub struct ModuleDescr {
    pub submodules: Vec<Submodule>
}

impl ModuleDescr {
    pub fn new(root: ast::Root) -> ModuleDescr {
        let submodules = root
            .modules()
            .filter_map(|module| {
                let name = module.name()?.text();
                if !module.has_semi() {
                    return None;
                }
                Some(Submodule { name })
            }).collect();

        ModuleDescr { submodules } }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Submodule {
    pub name: SmolStr,
}
