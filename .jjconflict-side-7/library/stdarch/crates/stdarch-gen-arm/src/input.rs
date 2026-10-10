use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{
    context::{self, GlobalContext},
    intrinsic::Intrinsic,
    predicate_forms::{PredicateForm, PredicationMask, PredicationMethods},
    typekinds::TypeKind,
    wildstring::WildString,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputType {
    /// PredicateForm variant argument
    #[serde(skip)] // Predicate forms have their own dedicated deserialization field. Skip.
    PredicateForm(PredicateForm),
    /// Operand from which to generate an N variant
    #[serde(skip)]
    NVariantOp(Option<WildString>),
    /// TypeKind variant argument
    Type(TypeKind),
}

impl InputType {
    /// Optionally unwraps as a PredicateForm.
    pub fn predicate_form(&self) -> Option<&PredicateForm> {
        match self {
            InputType::PredicateForm(pf) => Some(pf),
            _ => None,
        }
    }

    /// Optionally unwraps as a mutable PredicateForm
    pub fn predicate_form_mut(&mut self) -> Option<&mut PredicateForm> {
        match self {
            InputType::PredicateForm(pf) => Some(pf),
            _ => None,
        }
    }

    /// Optionally unwraps as a TypeKind.
    pub fn typekind(&self) -> Option<&TypeKind> {
        match self {
            InputType::Type(ty) => Some(ty),
            _ => None,
        }
    }

    /// Optionally unwraps as a NVariantOp
    pub fn n_variant_op(&self) -> Option<&WildString> {
        match self {
            InputType::NVariantOp(Some(op)) => Some(op),
            _ => None,
        }
    }
}

impl PartialOrd for InputType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InputType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;

        match (self, other) {
            (InputType::PredicateForm(pf1), InputType::PredicateForm(pf2)) => pf1.cmp(pf2),
            (InputType::Type(ty1), InputType::Type(ty2)) => ty1.cmp(ty2),

            (InputType::NVariantOp(None), InputType::NVariantOp(Some(..))) => Less,
            (InputType::NVariantOp(Some(..)), InputType::NVariantOp(None)) => Greater,
            (InputType::NVariantOp(_), InputType::NVariantOp(_)) => Equal,

            (InputType::Type(..), InputType::PredicateForm(..)) => Less,
            (InputType::PredicateForm(..), InputType::Type(..)) => Greater,

            (InputType::Type(..), InputType::NVariantOp(..)) => Less,
            (InputType::NVariantOp(..), InputType::Type(..)) => Greater,

            (InputType::PredicateForm(..), InputType::NVariantOp(..)) => Less,
            (InputType::NVariantOp(..), InputType::PredicateForm(..)) => Greater,
        }
    }
}

mod many_or_one {
    use serde::{Deserialize, Serialize, de::Deserializer, ser::Serializer};

    pub fn serialize<T, S>(vec: &Vec<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        if vec.len() == 1 {
            vec.first().unwrap().serialize(serializer)
        } else {
            vec.serialize(serializer)
        }
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(untagged)]
        enum ManyOrOne<T> {
            Many(Vec<T>),
            One(T),
        }

        match ManyOrOne::deserialize(deserializer)? {
            ManyOrOne::Many(vec) => Ok(vec),
            ManyOrOne::One(val) => Ok(vec![val]),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InputSet(#[serde(with = "many_or_one")] Vec<InputType>);

impl InputSet {
    pub fn get(&self, idx: usize) -> Option<&InputType> {
        self.0.get(idx)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &InputType> + '_ {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut InputType> + '_ {
        self.0.iter_mut()
    }

    pub fn into_iter(self) -> impl Iterator<Item = InputType> + Clone {
        self.0.into_iter()
    }

    pub fn types_len(&self) -> usize {
        self.iter().filter_map(|arg| arg.typekind()).count()
    }

    pub fn typekind(&self, idx: Option<usize>) -> Option<TypeKind> {
        let types_len = self.types_len();
        self.get(idx.unwrap_or(0)).and_then(move |arg: &InputType| {
            if (idx.is_none() && types_len != 1) || (idx.is_some() && types_len == 1) {
                None
            } else {
                arg.typekind().cloned()
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSetEntry(#[serde(with = "many_or_one")] Vec<InputSet>);

impl InputSetEntry {
    pub fn new(input: Vec<InputSet>) -> Self {
        Self(input)
    }

    pub fn get(&self, idx: usize) -> Option<&InputSet> {
        self.0.get(idx)
    }
}

fn validate_types<'de, D>(deserializer: D) -> Result<Vec<InputSetEntry>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Vec<InputSetEntry> = Vec::deserialize(deserializer)?;

    let mut it = v.iter();
    if let Some(first) = it.next() {
        it.try_fold(first, |last, cur| {
            if last.0.len() == cur.0.len() {
                Ok(cur)
            } else {
                Err("the length of the InputSets and the product lists must match".to_string())
            }
        })
        .map_err(de::Error::custom)?;
    }

    Ok(v)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntrinsicInput {
    #[serde(default)]
    #[serde(deserialize_with = "validate_types")]
    pub types: Vec<InputSetEntry>,

    #[serde(flatten)]
    pub predication_methods: PredicationMethods,

    /// Generates a _n variant where the specified operand is a primitive type
    /// that requires conversion to an SVE one. The `{_n}` wildcard is required
    /// in the intrinsic's name, otherwise an error will be thrown.
    #[serde(default)]
    pub n_variant_op: WildString,
}

impl IntrinsicInput {
    /// Extracts all the possible variants as an iterator.
    pub fn variants(
        &self,
        intrinsic: &Intrinsic,
    ) -> context::Result<impl Iterator<Item = InputSet> + '_> {
        let mut top_product = vec![];

        if !self.types.is_empty() {
            top_product.push(
                self.types
                    .iter()
                    .flat_map(|ty_in| {
                        ty_in
                            .0
                            .iter()
                            .map(|v| v.clone().into_iter())
                            .multi_cartesian_product()
                    })
                    .collect_vec(),
            )
        }

        if let Ok(mask) = PredicationMask::try_from(&intrinsic.signature.name) {
            top_product.push(
                PredicateForm::compile_list(&mask, &self.predication_methods)?
                    .into_iter()
                    .map(|pf| vec![InputType::PredicateForm(pf)])
                    .collect_vec(),
            )
        }

        if !self.n_variant_op.is_empty() {
            top_product.push(vec![
                vec![InputType::NVariantOp(None)],
                vec![InputType::NVariantOp(Some(self.n_variant_op.to_owned()))],
            ])
        }

        let it = top_product
            .into_iter()
            .map(|v| v.into_iter())
            .multi_cartesian_product()
            .filter(|set| !set.is_empty())
            .map(|set| InputSet(set.into_iter().flatten().collect_vec()));
        Ok(it)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorInput {
    #[serde(flatten)]
    pub ctx: GlobalContext,
    pub intrinsics: Vec<Intrinsic>,
}

#[cfg(test)]
mod tests {
    use crate::{
        input::*,
        predicate_forms::{DontCareMethod, ZeroingMethod},
    };

    #[test]
    fn test_empty() {
        let str = r#"types: []"#;
        let input: IntrinsicInput = serde_yaml::from_str(str).expect("failed to parse");
        let mut variants = input.variants(&Intrinsic::default()).unwrap().into_iter();
        assert_eq!(variants.next(), None);
    }

    #[test]
    fn test_product() {
        let str = r#"types:
- [f64, f32]
- [i64, [f64, f32]]
"#;
        let input: IntrinsicInput = serde_yaml::from_str(str).expect("failed to parse");
        let mut intrinsic = Intrinsic::default();
        intrinsic.signature.name = "test_intrinsic{_mx}".parse().unwrap();
        let mut variants = input.variants(&intrinsic).unwrap().into_iter();
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("f64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::Merging),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("f64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::DontCare(DontCareMethod::AsMerging)),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::Type("f64".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::Merging),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::Type("f64".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::DontCare(DontCareMethod::AsMerging)),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::Merging),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::DontCare(DontCareMethod::AsMerging)),
            ])),
        );
        assert_eq!(variants.next(), None);
    }

    #[test]
    fn test_n_variant() {
        let str = r#"types:
- [f64, f32]
n_variant_op: op2
"#;
        let input: IntrinsicInput = serde_yaml::from_str(str).expect("failed to parse");
        let mut variants = input.variants(&Intrinsic::default()).unwrap().into_iter();
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("f64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::NVariantOp(None),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("f64".parse().unwrap()),
                InputType::Type("f32".parse().unwrap()),
                InputType::NVariantOp(Some("op2".parse().unwrap())),
            ]))
        );
        assert_eq!(variants.next(), None)
    }

    #[test]
    fn test_invalid_length() {
        let str = r#"types: [i32, [[u64], [u32]]]"#;
        serde_yaml::from_str::<IntrinsicInput>(str).expect_err("failure expected");
    }

    #[test]
    fn test_invalid_predication() {
        let str = "types: []";
        let input: IntrinsicInput = serde_yaml::from_str(str).expect("failed to parse");
        let mut intrinsic = Intrinsic::default();
        intrinsic.signature.name = "test_intrinsic{_mxz}".parse().unwrap();
        input
            .variants(&intrinsic)
            .map(|v| v.collect_vec())
            .expect_err("failure expected");
    }

    #[test]
    fn test_invalid_predication_mask() {
        "test_intrinsic{_mxy}"
            .parse::<WildString>()
            .expect_err("failure expected");
        "test_intrinsic{_}"
            .parse::<WildString>()
            .expect_err("failure expected");
    }

    #[test]
    fn test_zeroing_predication() {
        let str = r#"types: [i64]
zeroing_method: { drop: inactive }"#;
        let input: IntrinsicInput = serde_yaml::from_str(str).expect("failed to parse");
        let mut intrinsic = Intrinsic::default();
        intrinsic.signature.name = "test_intrinsic{_mxz}".parse().unwrap();
        let mut variants = input.variants(&intrinsic).unwrap();
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::Merging),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::DontCare(DontCareMethod::AsZeroing)),
            ]))
        );
        assert_eq!(
            variants.next(),
            Some(InputSet(vec![
                InputType::Type("i64".parse().unwrap()),
                InputType::PredicateForm(PredicateForm::Zeroing(ZeroingMethod::Drop {
                    drop: "inactive".parse().unwrap()
                })),
            ]))
        );
        assert_eq!(variants.next(), None)
    }
}
