use crate::middle::typetree::TypeTree;
use std::str::FromStr;

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum DiffMode {
    Inactive,
    Source,
    Forward,
    Reverse,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum DiffActivity {
    None,
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
}

impl FromStr for DiffActivity {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffActivity, ()> {
        match s {
            "None" => Ok(DiffActivity::None),
            "Active" => Ok(DiffActivity::Active),
            "Const" => Ok(DiffActivity::Const),
            "Duplicated" => Ok(DiffActivity::Duplicated),
            "DuplicatedNoNeed" => Ok(DiffActivity::DuplicatedNoNeed),
            _ => Err(()),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct AutoDiffAttrs {
    pub mode: DiffMode,
    pub ret_activity: DiffActivity,
    pub input_activity: Vec<DiffActivity>,
}

impl AutoDiffAttrs {
    pub fn inactive() -> Self {
        AutoDiffAttrs {
            mode: DiffMode::Inactive,
            ret_activity: DiffActivity::None,
            input_activity: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            _ => true,
        }
    }

    pub fn is_source(&self) -> bool {
        match self.mode {
            DiffMode::Source => true,
            _ => false,
        }
    }
    pub fn apply_autodiff(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            DiffMode::Source => false,
            _ => true,
        }
    }

    pub fn into_item(
        self,
        source: String,
        target: String,
        inputs: Vec<TypeTree>,
        output: TypeTree,
    ) -> AutoDiffItem {
        AutoDiffItem { source, target, inputs, output, attrs: self }
    }
}

#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct AutoDiffItem {
    pub source: String,
    pub target: String,
    pub attrs: AutoDiffAttrs,
    pub inputs: Vec<TypeTree>,
    pub output: TypeTree,
}
