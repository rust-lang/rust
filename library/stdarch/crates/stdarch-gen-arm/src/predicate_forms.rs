use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::fmt;
use std::str::FromStr;

use crate::context;
use crate::expression::{Expression, FnCall, IdentifierType};
use crate::intrinsic::Intrinsic;
use crate::typekinds::{ToRepr, TypeKind};
use crate::wildcards::Wildcard;
use crate::wildstring::WildString;

const ZEROING_SUFFIX: &str = "_z";
const MERGING_SUFFIX: &str = "_m";
const DONT_CARE_SUFFIX: &str = "_x";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ZeroingMethod {
    /// Drop the specified argument and replace it with a zeroinitializer
    Drop { drop: WildString },
    /// Apply zero selection to the specified variable when zeroing
    Select { select: WildString },
}

impl PartialOrd for ZeroingMethod {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ZeroingMethod {
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DontCareMethod {
    #[default]
    Inferred,
    AsZeroing,
    AsMerging,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct PredicationMethods {
    /// Zeroing method, if the zeroing predicate form is used
    #[serde(default)]
    pub zeroing_method: Option<ZeroingMethod>,
    /// Don't care method, if the don't care predicate form is used
    #[serde(default)]
    pub dont_care_method: DontCareMethod,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PredicateForm {
    /// Enables merging predicate form
    Merging,
    /// Enables "don't care" predicate form.
    DontCare(DontCareMethod),
    /// Enables zeroing predicate form. If LLVM zeroselection is performed, then
    /// set the `select` field to the variable that gets set. Otherwise set the
    /// `drop` field if the zeroinitializer replaces a predicate when merging.
    Zeroing(ZeroingMethod),
}

impl PredicateForm {
    pub fn get_suffix(&self) -> &'static str {
        match self {
            PredicateForm::Zeroing { .. } => ZEROING_SUFFIX,
            PredicateForm::Merging => MERGING_SUFFIX,
            PredicateForm::DontCare { .. } => DONT_CARE_SUFFIX,
        }
    }

    pub fn make_zeroinitializer(ty: &TypeKind) -> Expression {
        FnCall::new_expression(
            format!("svdup_n_{}", ty.acle_notation_repr())
                .parse()
                .unwrap(),
            vec![if ty.base_type().unwrap().is_float() {
                Expression::FloatConstant(0.0)
            } else {
                Expression::IntConstant(0)
            }],
        )
    }

    pub fn make_zeroselector(pg_var: WildString, op_var: WildString, ty: &TypeKind) -> Expression {
        FnCall::new_expression(
            format!("svsel_{}", ty.acle_notation_repr())
                .parse()
                .unwrap(),
            vec![
                Expression::Identifier(pg_var, IdentifierType::Variable),
                Expression::Identifier(op_var, IdentifierType::Variable),
                Self::make_zeroinitializer(ty),
            ],
        )
    }

    pub fn post_build(&self, intrinsic: &mut Intrinsic) -> context::Result {
        // Drop the argument
        match self {
            PredicateForm::Zeroing(ZeroingMethod::Drop { drop: drop_var }) => {
                intrinsic.signature.drop_argument(drop_var)?
            }
            PredicateForm::DontCare(DontCareMethod::AsZeroing) => {
                if let ZeroingMethod::Drop { drop } = intrinsic
                    .input
                    .predication_methods
                    .zeroing_method
                    .to_owned()
                    .ok_or_else(|| {
                        "DontCareMethod::AsZeroing without zeroing method.".to_string()
                    })?
                {
                    intrinsic.signature.drop_argument(&drop)?
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn infer_dont_care(mask: &PredicationMask, methods: &PredicationMethods) -> PredicateForm {
        let method = if methods.dont_care_method == DontCareMethod::Inferred {
            if mask.has_zeroing()
                && matches!(methods.zeroing_method, Some(ZeroingMethod::Drop { .. }))
            {
                DontCareMethod::AsZeroing
            } else {
                DontCareMethod::AsMerging
            }
        } else {
            methods.dont_care_method
        };

        PredicateForm::DontCare(method)
    }

    pub fn compile_list(
        mask: &PredicationMask,
        methods: &PredicationMethods,
    ) -> context::Result<Vec<PredicateForm>> {
        let mut forms = Vec::new();

        if mask.has_merging() {
            forms.push(PredicateForm::Merging)
        }

        if mask.has_dont_care() {
            forms.push(Self::infer_dont_care(mask, methods))
        }

        if mask.has_zeroing() {
            if let Some(method) = methods.zeroing_method.to_owned() {
                forms.push(PredicateForm::Zeroing(method))
            } else {
                return Err(
                    "cannot create a zeroing variant without a zeroing method specified!"
                        .to_string(),
                );
            }
        }

        Ok(forms)
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, DeserializeFromStr, SerializeDisplay,
)]
pub struct PredicationMask {
    /// Merging
    m: bool,
    /// Don't care
    x: bool,
    /// Zeroing
    z: bool,
}

impl PredicationMask {
    pub fn has_merging(&self) -> bool {
        self.m
    }

    pub fn has_dont_care(&self) -> bool {
        self.x
    }

    pub fn has_zeroing(&self) -> bool {
        self.z
    }
}

impl FromStr for PredicationMask {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::default();
        for kind in s.bytes() {
            match kind {
                b'm' => result.m = true,
                b'x' => result.x = true,
                b'z' => result.z = true,
                _ => {
                    return Err(format!(
                        "unknown predicate form modifier: {}",
                        char::from(kind)
                    ));
                }
            }
        }

        if result.m || result.x || result.z {
            Ok(result)
        } else {
            Err("invalid predication mask".to_string())
        }
    }
}

impl fmt::Display for PredicationMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.m.then(|| write!(f, "m")).transpose()?;
        self.x.then(|| write!(f, "x")).transpose()?;
        self.z.then(|| write!(f, "z")).transpose().map(|_| ())
    }
}

impl TryFrom<&WildString> for PredicationMask {
    type Error = String;

    fn try_from(value: &WildString) -> Result<Self, Self::Error> {
        value
            .wildcards()
            .find_map(|w| {
                if let Wildcard::PredicateForms(mask) = w {
                    Some(*mask)
                } else {
                    None
                }
            })
            .ok_or_else(|| "no predicate forms were specified in the name".to_string())
    }
}
