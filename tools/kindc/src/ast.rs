#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub file: String,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub version: Option<u32>,
    pub declarations: Vec<KindDecl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindDecl {
    pub doc: Option<String>,
    pub name: DottedName,
    pub type_params: Vec<String>,
    pub body: Option<KindBody>,
    pub span: Span,
}

pub type DottedName = Vec<String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KindBody {
    Struct(Vec<FieldDecl>),
    Enum(Vec<VariantDecl>),
    Alias(TypeExpr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDecl {
    pub doc: Option<String>,
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDecl {
    pub doc: Option<String>,
    pub name: String,
    pub payload: VariantPayload,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantPayload {
    Unit,
    Tuple(TypeExpr),
    Struct(Vec<FieldDecl>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeExpr {
    pub name: DottedName,
    pub args: Vec<TypeExpr>,
    pub span: Span,
}
