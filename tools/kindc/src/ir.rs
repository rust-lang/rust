use indexmap::IndexMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub version: u32,
    pub kinds: IndexMap<String, ResolvedKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedKind {
    pub canonical_name: String,
    pub doc: Option<String>,
    pub kind_id: [u8; 16],
    pub shape: KindShape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KindShape {
    Builtin(BuiltinKind),
    Struct(Vec<ResolvedField>),
    Enum(Vec<ResolvedVariant>),
    Alias(ResolvedType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinKind {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    String,
    Bytes,
    Option,
    Result,
    List,
    Ref,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedField {
    pub name: String,
    pub doc: Option<String>,
    pub ty: ResolvedType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedVariant {
    pub name: String,
    pub doc: Option<String>,
    pub payload: ResolvedVariantPayload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedVariantPayload {
    Unit,
    Tuple(ResolvedType),
    Struct(Vec<ResolvedField>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedType {
    pub kind_ref: String, // Canonical name
    pub args: Vec<ResolvedType>,
}
