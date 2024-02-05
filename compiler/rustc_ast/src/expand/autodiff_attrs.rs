use crate::expand::typetree::TypeTree;
use std::str::FromStr;
use thin_vec::ThinVec;

use crate::NestedMetaItem;

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum DiffMode {
    Inactive,
    Source,
    Forward,
    Reverse,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum DiffActivity {
    None,
    Active,
    Const,
    Dual,
    DualNoNeed,
    Duplicated,
    DuplicatedNoNeed,
}

impl FromStr for DiffMode {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffMode, ()> {
        match s {
            "Inactive" => Ok(DiffMode::Inactive),
            "Source" => Ok(DiffMode::Source),
            "Forward" => Ok(DiffMode::Forward),
            "Reverse" => Ok(DiffMode::Reverse),
            _ => Err(()),
        }
    }
}
impl FromStr for DiffActivity {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffActivity, ()> {
        match s {
            "None" => Ok(DiffActivity::None),
            "Active" => Ok(DiffActivity::Active),
            "Const" => Ok(DiffActivity::Const),
            "Dual" => Ok(DiffActivity::Dual),
            "DualNoNeed" => Ok(DiffActivity::DualNoNeed),
            "Duplicated" => Ok(DiffActivity::Duplicated),
            "DuplicatedNoNeed" => Ok(DiffActivity::DuplicatedNoNeed),
            _ => Err(()),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AutoDiffAttrs {
    pub mode: DiffMode,
    pub ret_activity: DiffActivity,
    pub input_activity: Vec<DiffActivity>,
}

fn first_ident(x: &NestedMetaItem) -> rustc_span::symbol::Ident {
    let segments = &x.meta_item().unwrap().path.segments;
    assert!(segments.len() == 1);
    segments[0].ident
}
fn name(x: &NestedMetaItem) -> String {
    first_ident(x).name.to_string()
}

impl AutoDiffAttrs {
    pub fn has_ret_activity(&self) -> bool {
        match self.ret_activity {
            DiffActivity::None => false,
            _ => true,
        }
    }
    pub fn from_ast(meta_item: &ThinVec<NestedMetaItem>, has_ret: bool) -> Self {
        let mode = name(&meta_item[1]);
        let mode = DiffMode::from_str(&mode).unwrap();
        let activities: Vec<DiffActivity> = meta_item[2..]
            .iter()
            .map(|x| {
                let activity_str = name(&x);
                DiffActivity::from_str(&activity_str).unwrap()
            })
            .collect();

        // If a return type exist, we need to split the last activity,
        // otherwise we return None as placeholder.
        let (ret_activity, input_activity) = if has_ret {
            activities.split_last().unwrap()
        } else {
            (&DiffActivity::None, activities.as_slice())
        };

        AutoDiffAttrs { mode, ret_activity: *ret_activity, input_activity: input_activity.to_vec() }
    }
}

impl AutoDiffAttrs {
    pub fn inactive() -> Self {
        AutoDiffAttrs {
            mode: DiffMode::Inactive,
            ret_activity: DiffActivity::None,
            input_activity: Vec::new(),
        }
    }
    pub fn source() -> Self {
        AutoDiffAttrs {
            mode: DiffMode::Source,
            ret_activity: DiffActivity::None,
            input_activity: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            _ => {
                dbg!(&self);
                true
            }
        }
    }

    pub fn is_source(&self) -> bool {
        dbg!(&self);
        match self.mode {
            DiffMode::Source => true,
            _ => false,
        }
    }
    pub fn apply_autodiff(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            DiffMode::Source => false,
            _ => {
                dbg!(&self);
                true
            }
        }
    }

    pub fn into_item(
        self,
        source: String,
        target: String,
        inputs: Vec<TypeTree>,
        output: TypeTree,
    ) -> AutoDiffItem {
        dbg!(&self);
        AutoDiffItem { source, target, inputs, output, attrs: self }
    }
}

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AutoDiffItem {
    pub source: String,
    pub target: String,
    pub attrs: AutoDiffAttrs,
    pub inputs: Vec<TypeTree>,
    pub output: TypeTree,
}
