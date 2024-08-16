use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::expand::typetree::TypeTree;
use crate::expand::{Decodable, Encodable, HashStable_Generic};
use crate::ptr::P;
use crate::{Ty, TyKind};

#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum DiffMode {
    Inactive,
    Source,
    Forward,
    Reverse,
    ForwardFirst,
    ReverseFirst,
}

pub fn is_rev(mode: DiffMode) -> bool {
    match mode {
        DiffMode::Reverse | DiffMode::ReverseFirst => true,
        _ => false,
    }
}
pub fn is_fwd(mode: DiffMode) -> bool {
    match mode {
        DiffMode::Forward | DiffMode::ForwardFirst => true,
        _ => false,
    }
}

impl Display for DiffMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DiffMode::Inactive => write!(f, "Inactive"),
            DiffMode::Source => write!(f, "Source"),
            DiffMode::Forward => write!(f, "Forward"),
            DiffMode::Reverse => write!(f, "Reverse"),
            DiffMode::ForwardFirst => write!(f, "ForwardFirst"),
            DiffMode::ReverseFirst => write!(f, "ReverseFirst"),
        }
    }
}

pub fn valid_ret_activity(mode: DiffMode, activity: DiffActivity) -> bool {
    if activity == DiffActivity::None {
        // Only valid if primal returns (), but we can't check that here.
        return true;
    }
    match mode {
        DiffMode::Inactive => false,
        DiffMode::Source => false,
        DiffMode::Forward | DiffMode::ForwardFirst => {
            activity == DiffActivity::Dual
                || activity == DiffActivity::DualOnly
                || activity == DiffActivity::Const
        }
        DiffMode::Reverse | DiffMode::ReverseFirst => {
            activity == DiffActivity::Const
                || activity == DiffActivity::Active
                || activity == DiffActivity::ActiveOnly
        }
    }
}
fn is_ptr_or_ref(ty: &Ty) -> bool {
    match ty.kind {
        TyKind::Ptr(_) | TyKind::Ref(_, _) => true,
        _ => false,
    }
}
// TODO We should make this more robust to also
// accept aliases of f32 and f64
//fn is_float(ty: &Ty) -> bool {
//    false
//}
pub fn valid_ty_for_activity(ty: &P<Ty>, activity: DiffActivity) -> bool {
    if is_ptr_or_ref(ty) {
        return activity == DiffActivity::Dual
            || activity == DiffActivity::DualOnly
            || activity == DiffActivity::Duplicated
            || activity == DiffActivity::DuplicatedOnly
            || activity == DiffActivity::Const;
    }
    true
    //if is_scalar_ty(&ty) {
    //    return activity == DiffActivity::Active || activity == DiffActivity::ActiveOnly ||
    //        activity == DiffActivity::Const;
    //}
}
pub fn valid_input_activity(mode: DiffMode, activity: DiffActivity) -> bool {
    return match mode {
        DiffMode::Inactive => false,
        DiffMode::Source => false,
        DiffMode::Forward | DiffMode::ForwardFirst => {
            // These are the only valid cases
            activity == DiffActivity::Dual
                || activity == DiffActivity::DualOnly
                || activity == DiffActivity::Const
        }
        DiffMode::Reverse | DiffMode::ReverseFirst => {
            // These are the only valid cases
            activity == DiffActivity::Active
                || activity == DiffActivity::ActiveOnly
                || activity == DiffActivity::Const
                || activity == DiffActivity::Duplicated
                || activity == DiffActivity::DuplicatedOnly
        }
    };
}
pub fn invalid_input_activities(mode: DiffMode, activity_vec: &[DiffActivity]) -> Option<usize> {
    for i in 0..activity_vec.len() {
        if !valid_input_activity(mode, activity_vec[i]) {
            return Some(i);
        }
    }
    None
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum DiffActivity {
    None,
    Const,
    Active,
    ActiveOnly,
    Dual,
    DualOnly,
    Duplicated,
    DuplicatedOnly,
    FakeActivitySize,
}

impl Display for DiffActivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffActivity::None => write!(f, "None"),
            DiffActivity::Const => write!(f, "Const"),
            DiffActivity::Active => write!(f, "Active"),
            DiffActivity::ActiveOnly => write!(f, "ActiveOnly"),
            DiffActivity::Dual => write!(f, "Dual"),
            DiffActivity::DualOnly => write!(f, "DualOnly"),
            DiffActivity::Duplicated => write!(f, "Duplicated"),
            DiffActivity::DuplicatedOnly => write!(f, "DuplicatedOnly"),
            DiffActivity::FakeActivitySize => write!(f, "FakeActivitySize"),
        }
    }
}

impl FromStr for DiffMode {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffMode, ()> {
        match s {
            "Inactive" => Ok(DiffMode::Inactive),
            "Source" => Ok(DiffMode::Source),
            "Forward" => Ok(DiffMode::Forward),
            "Reverse" => Ok(DiffMode::Reverse),
            "ForwardFirst" => Ok(DiffMode::ForwardFirst),
            "ReverseFirst" => Ok(DiffMode::ReverseFirst),
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
            "ActiveOnly" => Ok(DiffActivity::ActiveOnly),
            "Const" => Ok(DiffActivity::Const),
            "Dual" => Ok(DiffActivity::Dual),
            "DualOnly" => Ok(DiffActivity::DualOnly),
            "Duplicated" => Ok(DiffActivity::Duplicated),
            "DuplicatedOnly" => Ok(DiffActivity::DuplicatedOnly),
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

impl AutoDiffAttrs {
    pub fn has_ret_activity(&self) -> bool {
        match self.ret_activity {
            DiffActivity::None => false,
            _ => true,
        }
    }
    pub fn has_active_only_ret(&self) -> bool {
        match self.ret_activity {
            DiffActivity::ActiveOnly => true,
            _ => false,
        }
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

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AutoDiffItem {
    pub source: String,
    pub target: String,
    pub attrs: AutoDiffAttrs,
    pub inputs: Vec<TypeTree>,
    pub output: TypeTree,
}

impl fmt::Display for AutoDiffItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Differentiating {} -> {}", self.source, self.target)?;
        write!(f, " with attributes: {:?}", self.attrs)?;
        write!(f, " with inputs: {:?}", self.inputs)?;
        write!(f, " with output: {:?}", self.output)
    }
}
