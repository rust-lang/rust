use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::{iter, mem, slice};

use rustc_ast::visit::FnKind;
#[allow(unused_imports)]
use rustc_ast::{
    AngleBracketedArg, Block, Expr, FieldDef, FnDecl, FnRetTy, GenericArg, GenericArgs, Item, Pat,
    Path, VariantData,
};
#[allow(unused_imports)]
use rustc_parse::parser::daikon_strs::{
    BOOL, CHAR, F32, F64, I8, I16, I32, I64, I128, ISIZE, STR, STRING, U8, U16, U32, U64, U128,
    UNIT, USIZE, VEC,
};
use rustc_parse::parser::item::{DO_VISITOR, OUTPUT_NAME};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{LazyLock, Mutex};
use thin_vec::ThinVec;

use rustc_ast::mut_visit::*;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{self, AssocCtxt, Visitor, VisitorResult, try_visit, walk_list};
use rustc_ast::{
    self as ast, AssocItemKind, AstNodeWrapper, AttrArgs, AttrStyle, AttrVec, DUMMY_NODE_ID,
    ExprKind, ForeignItemKind, HasAttrs, HasNodeId, Inline, ItemKind, MacStmtStyle, MetaItemInner,
    MetaItemKind, ModKind, NodeId, PatKind, StmtKind, TyKind, token,
};
use rustc_ast_pretty::pprust;
use rustc_attr_parsing::{AttributeParser, Early, EvalConfigResult, ShouldEmit, validate_attr};
use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::PResult;
use rustc_feature::Features;
use rustc_hir::Target;
use rustc_hir::def::MacroKinds;
use rustc_hir::limit::Limit;
use rustc_parse::parser::{
    AttemptLocalParseRecovery, CommaRecoveryMode, ForceCollect, Parser, RecoverColon, RecoverComma,
    token_descr,
};
use rustc_session::Session;
use rustc_session::lint::builtin::{UNUSED_ATTRIBUTES, UNUSED_DOC_COMMENTS};
use rustc_session::parse::feature_err;
use rustc_span::hygiene::SyntaxContext;
use rustc_span::{ErrorGuaranteed, FileName, Ident, LocalExpnId, Span, Symbol, sym};
use smallvec::SmallVec;

use crate::base::*;
use crate::config::{StripUnconfigured, attr_into_trace};
use crate::errors::{
    EmptyDelegationMac, GlobDelegationOutsideImpls, GlobDelegationTraitlessQpath, IncompleteParse,
    RecursionLimitReached, RemoveExprNotSupported, RemoveNodeNotSupported, UnsupportedKeyValue,
    WrongFragmentKind,
};
use crate::fluent_generated;
use crate::mbe::diagnostics::annotate_err_with_kind;
use crate::module::{
    DirOwnership, ParsedExternalMod, mod_dir_path, mod_file_path_from_attr, parse_external_mod,
};
use crate::placeholders::{PlaceholderExpander, placeholder};
use crate::stats::*;

macro_rules! ast_fragments {
    (
        $($Kind:ident($AstTy:ty) {
            $kind_name:expr;
            $(one
                fn $mut_visit_ast:ident;
                fn $visit_ast:ident;
                fn $ast_to_string:path;
            )?
            $(many
                fn $flat_map_ast_elt:ident;
                fn $visit_ast_elt:ident($($args:tt)*);
                fn $ast_to_string_elt:path;
            )?
            fn $make_ast:ident;
        })*
    ) => {
        /// A fragment of AST that can be produced by a single macro expansion.
        /// Can also serve as an input and intermediate result for macro expansion operations.
        pub enum AstFragment {
            OptExpr(Option<Box<ast::Expr>>),
            MethodReceiverExpr(Box<ast::Expr>),
            $($Kind($AstTy),)*
        }

        /// "Discriminant" of an AST fragment.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum AstFragmentKind {
            OptExpr,
            MethodReceiverExpr,
            $($Kind,)*
        }

        impl AstFragmentKind {
            pub fn name(self) -> &'static str {
                match self {
                    AstFragmentKind::OptExpr => "expression",
                    AstFragmentKind::MethodReceiverExpr => "expression",
                    $(AstFragmentKind::$Kind => $kind_name,)*
                }
            }

            fn make_from(self, result: Box<dyn MacResult + '_>) -> Option<AstFragment> {
                match self {
                    AstFragmentKind::OptExpr =>
                        result.make_expr().map(Some).map(AstFragment::OptExpr),
                    AstFragmentKind::MethodReceiverExpr =>
                        result.make_expr().map(AstFragment::MethodReceiverExpr),
                    $(AstFragmentKind::$Kind => result.$make_ast().map(AstFragment::$Kind),)*
                }
            }
        }

        impl AstFragment {
            fn add_placeholders(&mut self, placeholders: &[NodeId]) {
                if placeholders.is_empty() {
                    return;
                }
                match self {
                    $($(AstFragment::$Kind(ast) => ast.extend(placeholders.iter().flat_map(|id| {
                        ${ignore($flat_map_ast_elt)}
                        placeholder(AstFragmentKind::$Kind, *id, None).$make_ast()
                    })),)?)*
                    _ => panic!("unexpected AST fragment kind")
                }
            }

            pub(crate) fn make_opt_expr(self) -> Option<Box<ast::Expr>> {
                match self {
                    AstFragment::OptExpr(expr) => expr,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            }

            pub(crate) fn make_method_receiver_expr(self) -> Box<ast::Expr> {
                match self {
                    AstFragment::MethodReceiverExpr(expr) => expr,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            }

            $(pub fn $make_ast(self) -> $AstTy {
                match self {
                    AstFragment::$Kind(ast) => ast,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            })*

            fn make_ast<T: InvocationCollectorNode>(self) -> T::OutputTy {
                T::fragment_to_output(self)
            }

            pub(crate) fn mut_visit_with(&mut self, vis: &mut impl MutVisitor) {
                match self {
                    AstFragment::OptExpr(opt_expr) => {
                        if let Some(expr) = opt_expr.take() {
                            *opt_expr = vis.filter_map_expr(expr)
                        }
                    }
                    AstFragment::MethodReceiverExpr(expr) => vis.visit_method_receiver_expr(expr),
                    $($(AstFragment::$Kind(ast) => vis.$mut_visit_ast(ast),)?)*
                    $($(AstFragment::$Kind(ast) =>
                        ast.flat_map_in_place(|ast| vis.$flat_map_ast_elt(ast, $($args)*)),)?)*
                }
            }

            pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) -> V::Result {
                match self {
                    AstFragment::OptExpr(Some(expr)) => try_visit!(visitor.visit_expr(expr)),
                    AstFragment::OptExpr(None) => {}
                    AstFragment::MethodReceiverExpr(expr) => try_visit!(visitor.visit_method_receiver_expr(expr)),
                    $($(AstFragment::$Kind(ast) => try_visit!(visitor.$visit_ast(ast)),)?)*
                    $($(AstFragment::$Kind(ast) => walk_list!(visitor, $visit_ast_elt, &ast[..], $($args)*),)?)*
                }
                V::Result::output()
            }

            pub(crate) fn to_string(&self) -> String {
                match self {
                    AstFragment::OptExpr(Some(expr)) => pprust::expr_to_string(expr),
                    AstFragment::OptExpr(None) => unreachable!(),
                    AstFragment::MethodReceiverExpr(expr) => pprust::expr_to_string(expr),
                    $($(AstFragment::$Kind(ast) => $ast_to_string(ast),)?)*
                    $($(
                        AstFragment::$Kind(ast) => {
                            // The closure unwraps a `P` if present, or does nothing otherwise.
                            elems_to_string(&*ast, |ast| $ast_to_string_elt(&*ast))
                        }
                    )?)*
                }
            }
        }

        impl<'a> MacResult for crate::mbe::macro_rules::ParserAnyMacro<'a> {
            $(fn $make_ast(self: Box<crate::mbe::macro_rules::ParserAnyMacro<'a>>)
                           -> Option<$AstTy> {
                Some(self.make(AstFragmentKind::$Kind).$make_ast())
            })*
        }
    }
}

ast_fragments! {
    Expr(Box<ast::Expr>) {
        "expression";
        one fn visit_expr; fn visit_expr; fn pprust::expr_to_string;
        fn make_expr;
    }
    Pat(Box<ast::Pat>) {
        "pattern";
        one fn visit_pat; fn visit_pat; fn pprust::pat_to_string;
        fn make_pat;
    }
    Ty(Box<ast::Ty>) {
        "type";
        one fn visit_ty; fn visit_ty; fn pprust::ty_to_string;
        fn make_ty;
    }
    Stmts(SmallVec<[ast::Stmt; 1]>) {
        "statement";
        many fn flat_map_stmt; fn visit_stmt(); fn pprust::stmt_to_string;
        fn make_stmts;
    }
    Items(SmallVec<[Box<ast::Item>; 1]>) {
        "item";
        many fn flat_map_item; fn visit_item(); fn pprust::item_to_string;
        fn make_items;
    }
    TraitItems(SmallVec<[Box<ast::AssocItem>; 1]>) {
        "trait item";
        many fn flat_map_assoc_item; fn visit_assoc_item(AssocCtxt::Trait);
            fn pprust::assoc_item_to_string;
        fn make_trait_items;
    }
    ImplItems(SmallVec<[Box<ast::AssocItem>; 1]>) {
        "impl item";
        many fn flat_map_assoc_item; fn visit_assoc_item(AssocCtxt::Impl { of_trait: false });
            fn pprust::assoc_item_to_string;
        fn make_impl_items;
    }
    TraitImplItems(SmallVec<[Box<ast::AssocItem>; 1]>) {
        "impl item";
        many fn flat_map_assoc_item; fn visit_assoc_item(AssocCtxt::Impl { of_trait: true });
            fn pprust::assoc_item_to_string;
        fn make_trait_impl_items;
    }
    ForeignItems(SmallVec<[Box<ast::ForeignItem>; 1]>) {
        "foreign item";
        many fn flat_map_foreign_item; fn visit_foreign_item(); fn pprust::foreign_item_to_string;
        fn make_foreign_items;
    }
    Arms(SmallVec<[ast::Arm; 1]>) {
        "match arm";
        many fn flat_map_arm; fn visit_arm(); fn unreachable_to_string;
        fn make_arms;
    }
    ExprFields(SmallVec<[ast::ExprField; 1]>) {
        "field expression";
        many fn flat_map_expr_field; fn visit_expr_field(); fn unreachable_to_string;
        fn make_expr_fields;
    }
    PatFields(SmallVec<[ast::PatField; 1]>) {
        "field pattern";
        many fn flat_map_pat_field; fn visit_pat_field(); fn unreachable_to_string;
        fn make_pat_fields;
    }
    GenericParams(SmallVec<[ast::GenericParam; 1]>) {
        "generic parameter";
        many fn flat_map_generic_param; fn visit_generic_param(); fn unreachable_to_string;
        fn make_generic_params;
    }
    Params(SmallVec<[ast::Param; 1]>) {
        "function parameter";
        many fn flat_map_param; fn visit_param(); fn unreachable_to_string;
        fn make_params;
    }
    FieldDefs(SmallVec<[ast::FieldDef; 1]>) {
        "field";
        many fn flat_map_field_def; fn visit_field_def(); fn unreachable_to_string;
        fn make_field_defs;
    }
    Variants(SmallVec<[ast::Variant; 1]>) {
        "variant"; many fn flat_map_variant; fn visit_variant(); fn unreachable_to_string;
        fn make_variants;
    }
    WherePredicates(SmallVec<[ast::WherePredicate; 1]>) {
        "where predicate";
        many fn flat_map_where_predicate; fn visit_where_predicate(); fn unreachable_to_string;
        fn make_where_predicates;
    }
    Crate(ast::Crate) {
        "crate";
        one fn visit_crate; fn visit_crate; fn unreachable_to_string;
        fn make_crate;
    }
}

pub enum SupportsMacroExpansion {
    No,
    Yes { supports_inner_attrs: bool },
}

impl AstFragmentKind {
    pub(crate) fn dummy(self, span: Span, guar: ErrorGuaranteed) -> AstFragment {
        self.make_from(DummyResult::any(span, guar)).expect("couldn't create a dummy AST fragment")
    }

    pub fn supports_macro_expansion(self) -> SupportsMacroExpansion {
        match self {
            AstFragmentKind::OptExpr
            | AstFragmentKind::Expr
            | AstFragmentKind::MethodReceiverExpr
            | AstFragmentKind::Stmts
            | AstFragmentKind::Ty
            | AstFragmentKind::Pat => SupportsMacroExpansion::Yes { supports_inner_attrs: false },
            AstFragmentKind::Items
            | AstFragmentKind::TraitItems
            | AstFragmentKind::ImplItems
            | AstFragmentKind::TraitImplItems
            | AstFragmentKind::ForeignItems
            | AstFragmentKind::Crate => SupportsMacroExpansion::Yes { supports_inner_attrs: true },
            AstFragmentKind::Arms
            | AstFragmentKind::ExprFields
            | AstFragmentKind::PatFields
            | AstFragmentKind::GenericParams
            | AstFragmentKind::Params
            | AstFragmentKind::FieldDefs
            | AstFragmentKind::Variants
            | AstFragmentKind::WherePredicates => SupportsMacroExpansion::No,
        }
    }

    pub(crate) fn expect_from_annotatables(
        self,
        items: impl IntoIterator<Item = Annotatable>,
    ) -> AstFragment {
        let mut items = items.into_iter();
        match self {
            AstFragmentKind::Arms => {
                AstFragment::Arms(items.map(Annotatable::expect_arm).collect())
            }
            AstFragmentKind::ExprFields => {
                AstFragment::ExprFields(items.map(Annotatable::expect_expr_field).collect())
            }
            AstFragmentKind::PatFields => {
                AstFragment::PatFields(items.map(Annotatable::expect_pat_field).collect())
            }
            AstFragmentKind::GenericParams => {
                AstFragment::GenericParams(items.map(Annotatable::expect_generic_param).collect())
            }
            AstFragmentKind::Params => {
                AstFragment::Params(items.map(Annotatable::expect_param).collect())
            }
            AstFragmentKind::FieldDefs => {
                AstFragment::FieldDefs(items.map(Annotatable::expect_field_def).collect())
            }
            AstFragmentKind::Variants => {
                AstFragment::Variants(items.map(Annotatable::expect_variant).collect())
            }
            AstFragmentKind::WherePredicates => AstFragment::WherePredicates(
                items.map(Annotatable::expect_where_predicate).collect(),
            ),
            AstFragmentKind::Items => {
                AstFragment::Items(items.map(Annotatable::expect_item).collect())
            }
            AstFragmentKind::ImplItems => {
                AstFragment::ImplItems(items.map(Annotatable::expect_impl_item).collect())
            }
            AstFragmentKind::TraitImplItems => {
                AstFragment::TraitImplItems(items.map(Annotatable::expect_impl_item).collect())
            }
            AstFragmentKind::TraitItems => {
                AstFragment::TraitItems(items.map(Annotatable::expect_trait_item).collect())
            }
            AstFragmentKind::ForeignItems => {
                AstFragment::ForeignItems(items.map(Annotatable::expect_foreign_item).collect())
            }
            AstFragmentKind::Stmts => {
                AstFragment::Stmts(items.map(Annotatable::expect_stmt).collect())
            }
            AstFragmentKind::Expr => AstFragment::Expr(
                items.next().expect("expected exactly one expression").expect_expr(),
            ),
            AstFragmentKind::MethodReceiverExpr => AstFragment::MethodReceiverExpr(
                items.next().expect("expected exactly one expression").expect_expr(),
            ),
            AstFragmentKind::OptExpr => {
                AstFragment::OptExpr(items.next().map(Annotatable::expect_expr))
            }
            AstFragmentKind::Crate => {
                AstFragment::Crate(items.next().expect("expected exactly one crate").expect_crate())
            }
            AstFragmentKind::Pat | AstFragmentKind::Ty => {
                panic!("patterns and types aren't annotatable")
            }
        }
    }
}

pub struct Invocation {
    pub kind: InvocationKind,
    pub fragment_kind: AstFragmentKind,
    pub expansion_data: ExpansionData,
}

pub enum InvocationKind {
    Bang {
        mac: Box<ast::MacCall>,
        span: Span,
    },
    Attr {
        attr: ast::Attribute,
        /// Re-insertion position for inert attributes.
        pos: usize,
        item: Annotatable,
        /// Required for resolving derive helper attributes.
        derives: Vec<ast::Path>,
    },
    Derive {
        path: ast::Path,
        is_const: bool,
        item: Annotatable,
    },
    GlobDelegation {
        item: Box<ast::AssocItem>,
        /// Whether this is a trait impl or an inherent impl
        of_trait: bool,
    },
}

impl InvocationKind {
    fn placeholder_visibility(&self) -> Option<ast::Visibility> {
        // HACK: For unnamed fields placeholders should have the same visibility as the actual
        // fields because for tuple structs/variants resolve determines visibilities of their
        // constructor using these field visibilities before attributes on them are expanded.
        // The assumption is that the attribute expansion cannot change field visibilities,
        // and it holds because only inert attributes are supported in this position.
        match self {
            InvocationKind::Attr { item: Annotatable::FieldDef(field), .. }
            | InvocationKind::Derive { item: Annotatable::FieldDef(field), .. }
                if field.ident.is_none() =>
            {
                Some(field.vis.clone())
            }
            _ => None,
        }
    }
}

impl Invocation {
    pub fn span(&self) -> Span {
        match &self.kind {
            InvocationKind::Bang { span, .. } => *span,
            InvocationKind::Attr { attr, .. } => attr.span,
            InvocationKind::Derive { path, .. } => path.span,
            InvocationKind::GlobDelegation { item, .. } => item.span,
        }
    }

    fn span_mut(&mut self) -> &mut Span {
        match &mut self.kind {
            InvocationKind::Bang { span, .. } => span,
            InvocationKind::Attr { attr, .. } => &mut attr.span,
            InvocationKind::Derive { path, .. } => &mut path.span,
            InvocationKind::GlobDelegation { item, .. } => &mut item.span,
        }
    }
}

// Given a parameter pat, return its identifier name in a String
fn get_param_ident(pat: &Box<Pat>) -> String {
    match &pat.kind {
        PatKind::Ident(_mode, ident, None) => String::from(ident.as_str()),
        _ => panic!("Formal arg does not have simple identifier"),
    }
}

// Given a Rust type, return its "Java" type if there is a match
fn get_prim_rep_type(ty_str: &str) -> String {
    if ty_str == I8
        || ty_str == I16
        || ty_str == I32
        || ty_str == I64
        || ty_str == I128
        || ty_str == ISIZE
        || ty_str == U8
        || ty_str == U16
        || ty_str == U32
        || ty_str == U64
        || ty_str == U128
        || ty_str == USIZE
    {
        return String::from("int");
    } else if ty_str == F32 || ty_str == F64 {
        return String::from("");
    } else if ty_str == CHAR {
        return String::from("char");
    } else if ty_str == BOOL {
        return String::from("boolean");
    } else if ty_str == UNIT {
        return String::from("");
    } else if ty_str == STR || ty_str == STRING {
        return String::from("java.lang.String");
    }
    String::from("")
}

// Given the arguments to a Vec or array, return a RepType
// enum representing the Vec/array.
fn grok_vec_args(path: &Path) -> RepType {
    let mut is_ref = false;
    match &path.segments[path.segments.len() - 1].args {
        None => panic!("Vec args has no type name"),
        Some(args) => match &**args {
            GenericArgs::AngleBracketed(brack_args) => match &brack_args.args[0] {
                AngleBracketedArg::Arg(arg) => match &arg {
                    GenericArg::Type(arg_type) => {
                        match &get_rep_type(&arg_type.kind, &mut is_ref) {
                            RepType::Prim(arg_p_type) => RepType::PrimArray(arg_p_type.to_string()),
                            RepType::HashCodeStruct(struct_type) => {
                                RepType::HashCodeArray(struct_type.to_string())
                            }
                            _ => panic!("Multi-dim vec/array not supported"),
                        }
                    }
                    _ => panic!("Grok args failed 1"),
                },
                _ => panic!("Grok args failed 2"),
            },
            _ => panic!("Grok args failed 3"),
        },
    }
}

// Capable of representing the rep-type of a Rust type
// String payload represents the corresponding "Java" type
// i32 -> Prim("int")
// &[i32] -> PrimArray("int")
// [X; 2] -> HashCodeArray("X")
// &'a X -> HashCodeStruct("X")
#[derive(PartialEq)]
enum RepType {
    Prim(String),
    PrimArray(String),
    HashCodeArray(String),
    HashCodeStruct(String),
}

// Given a Rust type kind, return its RepType. Also note whether the type
// is a reference with is_ref.
fn get_rep_type(kind: &TyKind, is_ref: &mut bool) -> RepType {
    match &kind {
        TyKind::Array(arr_type, _) => match &get_rep_type(&arr_type.kind, is_ref) {
            RepType::Prim(p_type) => RepType::PrimArray(String::from(p_type)),
            RepType::HashCodeStruct(basic_type) => RepType::HashCodeArray(String::from(basic_type)),
            _ => panic!("higher-dim arrays not supported"),
        },
        TyKind::Slice(arr_type) => match &get_rep_type(&arr_type.kind, is_ref) {
            RepType::Prim(p_type) => RepType::PrimArray(String::from(p_type)),
            RepType::HashCodeStruct(basic_type) => RepType::HashCodeArray(String::from(basic_type)),
            _ => panic!("higher-dim arrays not supported"),
        },
        TyKind::Ptr(_) => todo!(),
        TyKind::Ref(_, mut_ty) => {
            *is_ref = true;
            return get_rep_type(&mut_ty.ty.kind, is_ref);
        }
        TyKind::Path(_, path) => {
            if path.segments.len() == 0 {
                panic!("Path has no type");
            }
            let ty_string = path.segments[path.segments.len() - 1].ident.as_str();
            let maybe_prim_rep = get_prim_rep_type(ty_string);
            if maybe_prim_rep != "" {
                return RepType::Prim(maybe_prim_rep);
            }
            if ty_string == VEC {
                // TODO
                return grok_vec_args(&path);
            }
            return RepType::HashCodeStruct(String::from(ty_string));
        }
        _ => todo!(),
    }
}

// Unused
#[allow(rustc::default_hash_types)]
fn map_params(decl: &Box<FnDecl>) -> HashMap<String, i32> {
    let mut res = HashMap::new();
    let mut i = 0;
    while i < decl.inputs.len() {
        res.insert(get_param_ident(&decl.inputs[i].pat), i as i32);
        i += 1;
    }
    res
}

// This struct is responsible for building a map from identifier to Struct
// This will not be needed once we do a first pass, we can read from a /tmp
// file to fill this purpose.
#[allow(rustc::default_hash_types)]
struct DeclsHashMapBuilder<'a> {
    pub map: &'a mut HashMap<String, Box<Item>>,
}

impl<'a> Visitor<'a> for DeclsHashMapBuilder<'a> {
    // Visit structs and fill hash map.
    fn visit_item(&mut self, item: &'a Item) {
        match &item.kind {
            ItemKind::Struct(ident, _, variant_data) => match variant_data {
                VariantData::Struct { fields: _, recovered: _ } => {
                    self.map.insert(String::from(ident.as_str()), Box::new(item.clone()));
                }
                VariantData::Tuple(_, _) => {}
                _ => {}
            },
            _ => {}
        }

        visit::walk_item(self, item);
    }
}

// Main struct for walking functions to write the decls file.
// map allows for quick retrieval of struct fields when a struct
// parameter is encountered.
// depth_limit tells us when to stop writing decls for recursive structs.
#[allow(rustc::default_hash_types)]
struct DaikonDeclsVisitor<'a> {
    pub map: &'a HashMap<String, Box<Item>>,
    pub depth_limit: u32,
}

// Represents a parameter or return value which must be written to decls.
// map: map from String to struct definition with field declarations
// var_name: parameter name, or "return" for return values.
// dec_type: Declared type of the value (dec-type for Daikon)
// rep_type: Rep type of the value (rep-type for Daikon)
// key: If the value is a struct, contains the struct type name for lookup,
//      otherwise None.
// field_decls: If the value is a struct, represents decl records for the
//              fields of this struct
// contents: If the value is Vec or array, a decls record for the contents
//           of this outer container.
// Note: it is maintained that only one of field_decls or contents will be Some.
#[allow(rustc::default_hash_types)]
struct TopLevlDecl<'a> {
    pub map: &'a HashMap<String, Box<Item>>,
    pub var_name: String,
    pub dec_type: String,
    pub rep_type: String,
    pub key: Option<String>, // struct name for looking up structs if this is a struct
    pub field_decls: Option<Vec<FieldDecl<'a>>>,
    pub contents: Option<ArrayContents<'a>>,
}

// Represents a field decl of a struct at some arb. depth.
// enclosing_var: the identifier of the struct which contains this field.
// field_name: name of this field
// See TopLevlDecl for other fields.
#[allow(rustc::default_hash_types)]
struct FieldDecl<'a> {
    pub map: &'a HashMap<String, Box<Item>>,
    pub var_name: String,
    pub dec_type: String,
    pub rep_type: String,
    pub enclosing_var: String,
    pub field_name: String,
    pub key: Option<String>,
    pub field_decls: Option<Vec<FieldDecl<'a>>>,
    pub contents: Option<ArrayContents<'a>>,
}

// Represents the array contents decl record (i.e., arr[..] or arr[..].g rather than arr)
// enclosing_var: name of the outer container for this array or Vec
// sub_contents: If we are an array of structs, we need ArrayContents for each field.
// See TopLevlDecl for other fields.
#[allow(rustc::default_hash_types)]
struct ArrayContents<'a> {
    pub map: &'a HashMap<String, Box<Item>>,
    pub var_name: String,
    pub dec_type: String,
    pub rep_type: String,
    pub enclosing_var: String,
    pub key: Option<String>,
    pub sub_contents: Option<Vec<ArrayContents<'a>>>, // only if this is a hashcode[], for printing subfield array records
}

impl<'a> ArrayContents<'a> {
    // Write out an ArrayContents to the decls file. We assume the cursor is at
    // the right spot and we simply append ourselves to the file.
    fn write(&mut self) {
        match &mut *DECLS.lock().unwrap() {
            None => panic!("Cannot open decls"),
            Some(decls) => {
                if self.var_name == "false" {
                    return;
                }

                writeln!(decls, "variable {}", self.var_name).ok();
                writeln!(decls, "  var-kind array").ok();
                writeln!(decls, "  enclosing-var {}", self.enclosing_var).ok();
                writeln!(decls, "  array 1").ok();
                writeln!(decls, "  dec-type {}", self.dec_type).ok();
                writeln!(decls, "  rep-type {}", self.rep_type).ok();
                writeln!(decls, "  comparability -1").ok();
            }
        }

        match &mut self.sub_contents {
            None => {}
            Some(sub_contents) => {
                let mut i = 0;
                while i < sub_contents.len() {
                    sub_contents[i].write();
                    i += 1;
                }
            }
        }
    }

    // If we are an array of structs, use our key to fetch field definitions
    // of our struct type.
    fn get_fields(&self, do_write: &mut bool) -> ThinVec<FieldDef> {
        // use self.key to look up who we are.
        match &self.key {
            None => panic!("No key for get_fields"),
            Some(key) => {
                let struct_item = self.map.get(key);
                match &struct_item {
                    None => {
                        // This is an Enum or Union or ?
                        *do_write = false;
                        ThinVec::new()
                    }
                    Some(struct_item) => match &struct_item.kind {
                        ItemKind::Struct(_, _, variant_data) => match variant_data {
                            VariantData::Struct { fields, recovered: _ } => fields.clone(),
                            _ => panic!("Struct is not VariantData::Struct"),
                        },
                        _ => panic!("struct_item is not a struct"),
                    },
                }
            }
        }
    }

    // If we are an array of structs, recursively populate sub_contents by creating
    // a new ArrayContents for each field.
    // do_write: I think this was a hack for avoiding structs/enums/unions which did
    //           not belong to the crate. That is again an ongoing issue with the /tmp
    //           file we need to create in the first pass.
    fn build_contents(&mut self, depth_limit: u32, do_write: &mut bool) {
        if depth_limit == 0 {
            return;
        }

        // fields of the struct in this array
        let fields = self.get_fields(do_write);
        if !*do_write {
            return;
        }

        let mut i = 0;
        while i < fields.len() {
            let field_name = match &fields[i].ident {
                Some(field_ident) => String::from(field_ident.as_str()),
                None => panic!("Field has no identifier"),
            };
            let var_name = format!("{}.{}", self.var_name, field_name);
            let mut is_ref = false;
            let mut do_write = true;
            let var_decl = match &get_rep_type(&fields[i].ty.kind, &mut is_ref) {
                RepType::Prim(p_type) => ArrayContents {
                    map: self.map,
                    var_name: var_name.clone(),
                    dec_type: format!("{}[]", p_type),
                    rep_type: format!("{}[]", p_type),
                    enclosing_var: self.var_name.clone(),
                    key: None,
                    sub_contents: None,
                },
                RepType::HashCodeStruct(ty_string) => {
                    let mut tmp = ArrayContents {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: format!("{}[]", ty_string),
                        rep_type: String::from("hashcode[]"),
                        enclosing_var: self.var_name.clone(),
                        key: Some(ty_string.clone()),
                        sub_contents: Some(Vec::new()),
                    };
                    tmp.build_contents(depth_limit - 1, &mut do_write);

                    // Error checking
                    if !do_write {
                        // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                        match &mut tmp.sub_contents {
                            None => panic!("Expected some field_decls 1"),
                            Some(sub_contents) => {
                                let mut j = 0;
                                while j < sub_contents.len() {
                                    sub_contents[j].var_name = String::from("false");
                                    j += 1;
                                }
                            }
                        }
                    }
                    if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                        // this record is also invalid
                        tmp.var_name = String::from("false");
                    }
                    tmp
                }
                RepType::PrimArray(_) => {
                    // only print pointers
                    ArrayContents {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: String::from("<higher-dim-array>"),
                        rep_type: String::from("hashcode[]"),
                        enclosing_var: self.var_name.clone(),
                        key: None, // we shouldn't be using this in write.
                        sub_contents: None,
                    }
                }
                RepType::HashCodeArray(_) => {
                    // only print pointers
                    ArrayContents {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: String::from("<higher-dim-array>"),
                        rep_type: String::from("hashcode[]"),
                        enclosing_var: self.var_name.clone(),
                        key: None,
                        sub_contents: None,
                    }
                }
            };
            match &mut self.sub_contents {
                None => panic!("No sub_contents in build_contents"),
                Some(sub_contents) => {
                    sub_contents.push(var_decl);
                }
            }

            i += 1;
        }
    }
}

impl<'a> FieldDecl<'a> {
    // Write this entire FieldDecl to the decls file.
    fn write(&mut self) {
        match &mut *DECLS.lock().unwrap() {
            None => panic!("Cannot open decls"),
            Some(decls) => {
                if self.var_name == "false" {
                    return;
                }

                writeln!(decls, "variable {}", self.var_name).ok();
                writeln!(decls, "  var-kind field {}", self.field_name).ok();
                writeln!(decls, "  enclosing-var {}", self.enclosing_var).ok();
                writeln!(decls, "  dec-type {}", self.dec_type).ok();
                writeln!(decls, "  rep-type {}", self.rep_type).ok();
                writeln!(decls, "  comparability -1").ok();
            }
        }

        match &mut self.field_decls {
            None => {}
            Some(field_decls) => {
                let mut i = 0;
                while i < field_decls.len() {
                    field_decls[i].write();
                    i += 1;
                }
                return;
            }
        }
        match &mut self.contents {
            None => {}
            Some(contents) => {
                contents.write();
            }
        }
    }

    // If we are a struct type field, use our key to get field definitions
    // for the struct type.
    fn get_fields(&self, do_write: &mut bool) -> ThinVec<FieldDef> {
        // use self.key to look up who we are.
        match &self.key {
            None => panic!("No key for get_fields"),
            Some(key) => {
                let struct_item = self.map.get(key);
                match &struct_item {
                    None => {
                        *do_write = false;
                        ThinVec::new()
                    }
                    Some(struct_item) => match &struct_item.kind {
                        ItemKind::Struct(_, _, variant_data) => match variant_data {
                            VariantData::Struct { fields, recovered: _ } => fields.clone(),
                            _ => panic!("Struct is not VariantData::Struct"),
                        },
                        _ => panic!("struct_item is not a struct"),
                    },
                }
            }
        }
    }

    // If we are a struct field, recursively build up our field_decls by
    // creating a new FieldDecl for each field.
    fn build_fields(&mut self, depth_limit: u32, do_write: &mut bool) {
        if depth_limit == 0 {
            // Invalidate ourselves for writing? Or will writing stop too...
            return;
        }

        let fields = self.get_fields(do_write);
        if !*do_write {
            return;
        }
        // do we really need error checking after this? maybe, cause you can have Vec<Struct> where Struct has an Enum field,

        let mut i = 0;
        while i < fields.len() {
            let field_name = match &fields[i].ident {
                Some(field_ident) => String::from(field_ident.as_str()),
                None => panic!("Field has no identifier"),
            };
            let var_name = format!("{}.{}", self.var_name, field_name);
            let mut is_ref = false;
            let mut do_write = true;
            let var_decl = match &get_rep_type(&fields[i].ty.kind, &mut is_ref) {
                RepType::Prim(p_type) => {
                    FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: p_type.clone(),
                        rep_type: p_type.clone(),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: None,
                        field_decls: None,
                        contents: None,
                    } // Ready to write.
                }
                RepType::HashCodeStruct(ty_string) => {
                    let mut tmp = FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: ty_string.clone(),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: Some(ty_string.clone()),
                        field_decls: Some(Vec::new()),
                        contents: None,
                    };
                    tmp.build_fields(depth_limit - 1, &mut do_write);

                    // Error checking
                    if !do_write {
                        // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                        match &mut tmp.field_decls {
                            None => panic!("Expected some field_decls 1"),
                            Some(field_decls) => {
                                let mut j = 0;
                                while j < field_decls.len() {
                                    field_decls[j].var_name = String::from("false");
                                    j += 1;
                                }
                            }
                        }
                    }
                    if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                        // this record is also invalid
                        tmp.var_name = String::from("false");
                    }
                    tmp
                }
                RepType::PrimArray(p_type) => {
                    FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: format!("{}[]", p_type),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: None,
                        field_decls: None,
                        contents: Some(ArrayContents {
                            map: self.map,
                            var_name: format!("{}[..]", var_name),
                            dec_type: format!("{}[]", p_type),
                            rep_type: format!("{}[]", p_type),
                            enclosing_var: var_name.clone(),
                            key: None,
                            sub_contents: None,
                        }), // Ready to write.
                    }
                }
                RepType::HashCodeArray(ty_string) => {
                    let mut tmp = FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: format!("{}[]", ty_string),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: Some(ty_string.clone()),
                        field_decls: None,
                        contents: Some(ArrayContents {
                            map: self.map,
                            var_name: format!("{}[..]", var_name),
                            dec_type: format!("{}[]", ty_string),
                            rep_type: String::from("hashcode[]"),
                            enclosing_var: var_name.clone(),
                            key: Some(ty_string.clone()),
                            sub_contents: Some(Vec::new()),
                        }),
                    };
                    match &mut tmp.contents {
                        None => panic!(""),
                        Some(contents) => {
                            contents.build_contents(depth_limit - 1, &mut do_write);

                            // Error checking
                            if !do_write {
                                // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                                match &mut contents.sub_contents {
                                    None => panic!("Expected some field_decls 1"),
                                    Some(sub_contents) => {
                                        let mut j = 0;
                                        while j < sub_contents.len() {
                                            sub_contents[j].var_name = String::from("false");
                                            j += 1;
                                        }
                                    }
                                }
                            }
                            if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                                // this record is also invalid
                                tmp.var_name = String::from("false");
                            }
                        }
                    }
                    tmp
                }
            };
            match &mut self.field_decls {
                None => panic!("No field_decls in build_fields"),
                Some(field_decls) => {
                    field_decls.push(var_decl);
                }
            }

            i += 1;
        }
    }
}

impl<'a> TopLevlDecl<'a> {
    // Write this entire TopLevlDecl to the decls file.
    fn write(&mut self) {
        match &mut *DECLS.lock().unwrap() {
            None => panic!("Cannot open decls"),
            Some(decls) => {
                if self.var_name == "false" {
                    return;
                }

                writeln!(decls, "variable {}", self.var_name).ok();
                writeln!(decls, "  var-kind variable").ok();
                writeln!(decls, "  dec-type {}", self.dec_type).ok();
                writeln!(decls, "  rep-type {}", self.rep_type).ok();
                writeln!(decls, "  flags is_param").ok();
                writeln!(decls, "  comparability -1").ok();
            }
        }

        match &mut self.field_decls {
            None => {}
            Some(field_decls) => {
                let mut i = 0;
                while i < field_decls.len() {
                    field_decls[i].write();
                    i += 1;
                }
                return;
            }
        }
        match &mut self.contents {
            None => {}
            Some(contents) => {
                contents.write();
            }
        }
    }

    // If we are a struct variable, recursively build declarations for our
    // fields. Almost or maybe exactly the same as FieldDecl::build_fields.
    fn build_fields(&mut self, depth_limit: u32, do_write: &mut bool) {
        if depth_limit == 0 {
            // Invalidate ourselves for writing? Or will writing stop too...
            return;
        }

        let fields = self.get_fields(do_write);
        if !*do_write {
            return;
        }

        let mut i = 0;
        while i < fields.len() {
            let field_name = match &fields[i].ident {
                Some(field_ident) => String::from(field_ident.as_str()),
                None => panic!("Field has no identifier"),
            };
            let var_name = format!("{}.{}", self.var_name, field_name);
            let mut is_ref = false;
            let mut do_write = true;
            let var_decl = match &get_rep_type(&fields[i].ty.kind, &mut is_ref) {
                RepType::Prim(p_type) => {
                    FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: p_type.clone(),
                        rep_type: p_type.clone(),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: None,
                        field_decls: None,
                        contents: None,
                    } // Ready to write.
                }
                RepType::HashCodeStruct(ty_string) => {
                    let mut tmp = FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: ty_string.clone(),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: Some(ty_string.clone()),
                        field_decls: Some(Vec::new()),
                        contents: None,
                    };
                    tmp.build_fields(depth_limit - 1, &mut do_write);

                    // Error checking
                    if !do_write {
                        // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                        match &mut tmp.field_decls {
                            None => panic!("Expected some field_decls 1"),
                            Some(field_decls) => {
                                let mut j = 0;
                                while j < field_decls.len() {
                                    field_decls[j].var_name = String::from("false");
                                    j += 1;
                                }
                            }
                        }
                    }
                    if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                        // this record is also invalid
                        tmp.var_name = String::from("false");
                    }
                    tmp
                }
                RepType::PrimArray(p_type) => {
                    FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: format!("{}[]", p_type),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: None,
                        field_decls: None,
                        contents: Some(ArrayContents {
                            map: self.map,
                            var_name: format!("{}[..]", var_name),
                            dec_type: format!("{}[]", p_type),
                            rep_type: format!("{}[]", p_type),
                            enclosing_var: var_name.clone(),
                            key: None,
                            sub_contents: None,
                        }), // Ready to write.
                    }
                }
                RepType::HashCodeArray(ty_string) => {
                    let mut tmp = FieldDecl {
                        map: self.map,
                        var_name: var_name.clone(),
                        dec_type: format!("{}[]", ty_string),
                        rep_type: String::from("hashcode"),
                        enclosing_var: self.var_name.clone(),
                        field_name: field_name.clone(),
                        key: Some(ty_string.clone()),
                        field_decls: None,
                        contents: Some(ArrayContents {
                            map: self.map,
                            var_name: format!("{}[..]", var_name),
                            dec_type: format!("{}[]", ty_string),
                            rep_type: String::from("hashcode[]"),
                            enclosing_var: var_name.clone(),
                            key: Some(ty_string.clone()),
                            sub_contents: Some(Vec::new()),
                        }),
                    };
                    match &mut tmp.contents {
                        None => panic!(""),
                        Some(contents) => {
                            contents.build_contents(depth_limit - 1, &mut do_write);

                            // Error checking
                            if !do_write {
                                // Any "fields" are invalid, but tmp could be an enum/union and pointers is valid.
                                match &mut contents.sub_contents {
                                    None => panic!("Expected some field_decls 1"),
                                    Some(sub_contents) => {
                                        let mut j = 0;
                                        while j < sub_contents.len() {
                                            sub_contents[j].var_name = String::from("false");
                                            j += 1;
                                        }
                                    }
                                }
                            }

                            if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                                // this record is also invalid
                                tmp.var_name = String::from("false");
                            }
                        }
                    }
                    tmp
                }
            };
            match &mut self.field_decls {
                None => panic!("No field_decls in build_fields"),
                Some(field_decls) => {
                    field_decls.push(var_decl);
                }
            }

            i += 1;
        }
    }

    // If we are a struct variable, use our key to get field definitions
    // for our struct type.
    fn get_fields(&self, do_write: &mut bool) -> ThinVec<FieldDef> {
        // use self.key to look up who we are.
        match &self.key {
            None => panic!("No key for get_fields"),
            Some(key) => {
                let struct_item = self.map.get(key);
                match &struct_item {
                    None => {
                        *do_write = false;
                        ThinVec::new()
                    }
                    Some(struct_item) => match &struct_item.kind {
                        ItemKind::Struct(_, _, variant_data) => match variant_data {
                            VariantData::Struct { fields, recovered: _ } => fields.clone(),
                            _ => panic!("Struct is not VariantData::Struct"),
                        },
                        _ => panic!("struct_item is not a struct"),
                    },
                }
            }
        }
    }
}

// Helper to write function entries into the decls file.
fn write_entry(ppt_name: String) {
    match &mut *DECLS.lock().unwrap() {
        None => panic!("Cannot access decls"),
        Some(decls) => {
            writeln!(decls, "ppt {}:::ENTER", ppt_name).ok();
            writeln!(decls, "ppt-type enter").ok();
        }
    }
}

// Helper to write function exits into the decls file.
fn write_exit(ppt_name: String, exit_counter: usize) {
    match &mut *DECLS.lock().unwrap() {
        None => panic!("Cannot access decls"),
        Some(decls) => {
            writeln!(decls, "ppt {}:::EXIT{}", ppt_name, exit_counter).ok();
            writeln!(decls, "ppt-type exit").ok();
        }
    }
}

// Helper to add a newline in the decls file.
fn write_newline() {
    match &mut *DECLS.lock().unwrap() {
        None => panic!("Cannot access decls"),
        Some(decls) => {
            writeln!(decls, "").ok();
        }
    }
}

// Helper to write metadata header into the decls file.
fn write_header() {
    match &mut *DECLS.lock().unwrap() {
        None => panic!("Cannot access decls"),
        Some(decls) => {
            writeln!(decls, "decl-version 2.0").ok();
            writeln!(decls, "input-language Rust").ok();
            writeln!(decls, "var-comparability implicit").ok();
        }
    }
}

impl<'a> DaikonDeclsVisitor<'a> {
    // Walk an if expression looking for returns.
    // See rustc_parse::parser::item::grok_expr_for_if.
    #[allow(rustc::default_hash_types)]
    fn grok_expr_for_if(
        &mut self,
        expr: &Box<Expr>,
        exit_counter: &mut usize,
        ppt_name: String,
        param_decls: &mut Vec<TopLevlDecl<'_>>,
        param_to_block_idx: &HashMap<String, i32>,
        ret_ty: &FnRetTy,
    ) {
        match &expr.kind {
            ExprKind::Block(block, _) => {
                self.grok_block(
                    ppt_name.clone(),
                    block,
                    param_decls,
                    &param_to_block_idx,
                    &ret_ty,
                    exit_counter,
                );
            }
            ExprKind::If(_, if_block, None) => {
                self.grok_block(
                    ppt_name.clone(),
                    if_block,
                    param_decls,
                    &param_to_block_idx,
                    &ret_ty,
                    exit_counter,
                );
            }
            ExprKind::If(_, if_block, Some(another_expr)) => {
                self.grok_block(
                    ppt_name.clone(),
                    if_block,
                    param_decls,
                    &param_to_block_idx,
                    &ret_ty,
                    exit_counter,
                );
                self.grok_expr_for_if(
                    another_expr,
                    exit_counter,
                    ppt_name.clone(),
                    param_decls,
                    &param_to_block_idx,
                    &ret_ty,
                );
            }
            _ => panic!("Internal error handling if stmt with else!"),
        }
    }

    // Process an entire stmt to identify an exit point or recurse on blocks.
    // See rustc_parse::parser::item::grok_stmt.
    #[allow(rustc::default_hash_types)]
    fn grok_stmt(
        &mut self,
        loc: usize,
        body: &Box<Block>,
        exit_counter: &mut usize,
        ppt_name: String,
        param_decls: &mut Vec<TopLevlDecl<'_>>,
        param_to_block_idx: &HashMap<String, i32>,
        ret_ty: &FnRetTy,
    ) -> usize {
        let mut i = loc;
        match &body.stmts[i].kind {
            StmtKind::Let(_local) => {
                return i + 1;
            }
            StmtKind::Item(_item) => {
                return i + 1;
            }
            StmtKind::Expr(no_semi_expr) => match &no_semi_expr.kind {
                // Blocks.
                // recurse on nested block,
                // but we still only grokked one (block) stmt, so just
                // move to the next stmt (return i+1)
                ExprKind::Block(block, _) => {
                    self.grok_block(
                        ppt_name.clone(),
                        block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );
                    return i + 1;
                }
                ExprKind::If(_, if_block, None) => {
                    // no else
                    self.grok_block(
                        ppt_name.clone(),
                        if_block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );
                    return i + 1;
                }
                ExprKind::If(_, if_block, Some(expr)) => {
                    // yes else
                    self.grok_block(
                        ppt_name.clone(),
                        if_block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );

                    self.grok_expr_for_if(
                        expr,
                        exit_counter,
                        ppt_name.clone(),
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                    );
                    return i + 1;
                }
                ExprKind::While(_, while_block, _) => {
                    self.grok_block(
                        ppt_name.clone(),
                        while_block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );
                    return i + 1;
                }
                ExprKind::ForLoop { pat: _, iter: _, body: for_block, label: _, kind: _ } => {
                    self.grok_block(
                        ppt_name.clone(),
                        for_block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );
                    return i + 1;
                }
                ExprKind::Loop(loop_block, _, _) => {
                    self.grok_block(
                        ppt_name.clone(),
                        loop_block,
                        param_decls,
                        &param_to_block_idx,
                        &ret_ty,
                        exit_counter,
                    );
                    return i + 1;
                } // missing Match blocks, TryBlock, Const block? probably more
                _ => {}
            },
            // Look for returns. dtrace passes have run, so all exit points should
            // be identifiable by an explicit return stmt.
            StmtKind::Semi(semi) => match &semi.kind {
                ExprKind::Ret(None) => {
                    write_exit(ppt_name.clone(), *exit_counter);
                    *exit_counter += 1;
                    let mut idx = 0;
                    while idx < param_decls.len() {
                        param_decls[idx].write();
                        idx += 1;
                    }
                    write_newline();

                    // we're sitting on the void return we just processed, so inc
                    // to move on
                    i += 1;
                }
                ExprKind::Ret(Some(_)) => {
                    write_exit(ppt_name.clone(), *exit_counter);
                    *exit_counter += 1;
                    let mut idx = 0;
                    while idx < param_decls.len() {
                        param_decls[idx].write();
                        idx += 1;
                    }

                    // make return TopLevlDecl
                    match &ret_ty {
                        FnRetTy::Default(_) => {} // no return record to be had.
                        FnRetTy::Ty(ty) => {
                            let var_name = String::from("return");
                            let mut is_ref = false;
                            let mut do_write = true;
                            let mut return_decl = match &get_rep_type(&ty.kind, &mut is_ref) {
                                RepType::Prim(p_type) => {
                                    TopLevlDecl {
                                        map: self.map,
                                        var_name: var_name.clone(),
                                        dec_type: p_type.clone(),
                                        rep_type: p_type.clone(),
                                        key: None,
                                        field_decls: None,
                                        contents: None,
                                    } // Ready to write this var decl.
                                }
                                RepType::HashCodeStruct(ty_string) => {
                                    // do_write = !ty_string.starts_with("Option") && !ty_string.starts_with("Result");
                                    // println!("do_write is {} for {}", do_write, ty_string);
                                    let mut tmp = TopLevlDecl {
                                        map: self.map,
                                        var_name: var_name.clone(),
                                        dec_type: ty_string.clone(),
                                        rep_type: String::from("hashcode"),
                                        key: Some(ty_string.clone()),
                                        field_decls: Some(Vec::new()),
                                        contents: None,
                                    };
                                    tmp.build_fields(self.depth_limit, &mut do_write);

                                    // Error checking
                                    if !do_write {
                                        // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                                        match &mut tmp.field_decls {
                                            None => panic!("Expected some field_decls 1"),
                                            Some(field_decls) => {
                                                let mut j = 0;
                                                while j < field_decls.len() {
                                                    field_decls[j].var_name = String::from("false");
                                                    j += 1;
                                                }
                                            }
                                        }
                                    }
                                    if ty_string.starts_with("Option")
                                        || ty_string.starts_with("Result")
                                    {
                                        // this record is also invalid
                                        tmp.var_name = String::from("false");
                                    }
                                    tmp
                                }
                                RepType::PrimArray(p_type) => {
                                    TopLevlDecl {
                                        map: self.map,
                                        var_name: var_name.clone(),
                                        dec_type: format!("{}[]", p_type),
                                        rep_type: String::from("hashcode"),
                                        key: None,
                                        field_decls: None,
                                        contents: Some(ArrayContents {
                                            map: self.map,
                                            var_name: format!("{}[..]", var_name),
                                            dec_type: format!("{}[]", p_type),
                                            rep_type: format!("{}[]", p_type),
                                            enclosing_var: var_name.clone(),
                                            key: None,
                                            sub_contents: None,
                                        }), // Ready to write this var_decl.
                                    }
                                }
                                RepType::HashCodeArray(ty_string) => {
                                    let mut tmp = TopLevlDecl {
                                        map: self.map,
                                        var_name: var_name.clone(),
                                        dec_type: format!("{}[]", ty_string),
                                        rep_type: String::from("hashcode"),
                                        key: Some(ty_string.clone()),
                                        field_decls: None,
                                        contents: Some(ArrayContents {
                                            map: self.map,
                                            var_name: format!("{}[..]", var_name),
                                            dec_type: format!("{}[]", ty_string),
                                            rep_type: String::from("hashcode[]"),
                                            enclosing_var: var_name.clone(),
                                            key: Some(ty_string.clone()),
                                            sub_contents: Some(Vec::new()),
                                        }),
                                    };
                                    match &mut tmp.contents {
                                        None => panic!(""),
                                        Some(contents) => {
                                            contents.build_contents(
                                                self.depth_limit - 1,
                                                &mut do_write,
                                            );

                                            // Error checking
                                            if !do_write {
                                                // Any "fields" are invalid, but tmp could be an enum/union and pointers is valid.
                                                match &mut contents.sub_contents {
                                                    None => panic!("Expected some field_decls 1"),
                                                    Some(sub_contents) => {
                                                        let mut j = 0;
                                                        while j < sub_contents.len() {
                                                            sub_contents[j].var_name =
                                                                String::from("false");
                                                            j += 1;
                                                        }
                                                    }
                                                }
                                            }

                                            if ty_string.starts_with("Option")
                                                || ty_string.starts_with("Result")
                                            {
                                                // this record is also invalid
                                                tmp.var_name = String::from("false");
                                            }
                                        }
                                    }
                                    tmp
                                }
                            };
                            return_decl.write();
                        }
                    }

                    write_newline();
                    // probably:
                    i += 1;
                }
                ExprKind::Call(_call, _params) => {
                    return i + 1;
                } // Maybe check for drop and other invalidations
                _ => {
                    return i + 1;
                } // other things you overlooked
            },
            // StmtKind::Expr(no_semi_expr) => match &no_semi_expr.kind {
            //     ExprKind::Match(..) => {
            //         return i + 1;
            //     }
            //     _ => panic!("is this non-semi expr a return or a valid non-semi expr?"),
            // },
            _ => {
                return i + 1;
            }
        }
        i
    }

    // Walk a new block looking for exit points and nested blocks.
    // See rustc_parse::parser::item::grok_block.
    #[allow(rustc::default_hash_types)]
    fn grok_block(
        &mut self,
        ppt_name: String,
        body: &Box<Block>,
        param_decls: &mut Vec<TopLevlDecl<'_>>,
        param_to_block_idx: &HashMap<String, i32>,
        ret_ty: &FnRetTy,
        exit_counter: &mut usize,
    ) {
        let mut i = 0;

        // assuming no unreachable statements.
        while i < body.stmts.len() {
            // make sure loop bound is growing as we insert stmts
            i = self.grok_stmt(
                i,
                body,
                exit_counter,
                ppt_name.clone(),
                param_decls,
                &param_to_block_idx,
                &ret_ty,
            ); // match on Semi and blocks mainly for now, find return <expr>; and add an exit point.
        }
    }

    // is it a good idea to store which params are valid at each exit
    // ppt for the decls pass which happens after this?
    // then the decls pass just needs to
    // 1: visit_item to build HashMap<ident, StructNode>
    // 2: visit_fn, grok sig, and grok exit ppts using structural
    //    recursion on StructNodes for nesting. Need to use depth counter
    //    for a base case.

    // Walk a function body looking for exit points.
    // See rustc_parse::parser::item::grok_fn_body.
    #[allow(rustc::default_hash_types)]
    fn grok_fn_body(
        &mut self,
        ppt_name: String,
        body: &Box<Block>,
        param_decls: &mut Vec<TopLevlDecl<'_>>,
        param_to_block_idx: HashMap<String, i32>,
        ret_ty: &FnRetTy,
    ) {
        // look for returns and nested blocks (recurse in those cases)
        let mut exit_counter = 1;

        // assuming no unreachable statements.
        let mut i = 0;
        while i < body.stmts.len() {
            // make sure loop bound is growing as we insert stmts
            i = self.grok_stmt(
                i,
                body,
                &mut exit_counter,
                ppt_name.clone(),
                param_decls,
                &param_to_block_idx,
                &ret_ty,
            );
        }
    }
}

// Process a function signature and build up a new Vec<TopLevlDecl>
// ready to be subsequently written to the decls file before we
// walk the function body looking for exit points.
// See rustc_parse::parser::item::grok_fn_sig.
#[allow(rustc::default_hash_types)]
fn grok_fn_sig<'a>(
    decl: &'a Box<FnDecl>,
    map: &'a HashMap<String, Box<Item>>,
    depth_limit: u32,
) -> Vec<TopLevlDecl<'a>> {
    let mut var_decls: Vec<TopLevlDecl<'_>> = Vec::new();
    let mut i = 0;
    while i < decl.inputs.len() {
        let var_name = get_param_ident(&decl.inputs[i].pat);
        let mut is_ref = false;
        let mut do_write = true;
        let toplevl_decl = match &get_rep_type(&decl.inputs[i].ty.kind, &mut is_ref) {
            RepType::Prim(p_type) => {
                TopLevlDecl {
                    map,
                    var_name: var_name.clone(),
                    dec_type: p_type.clone(),
                    rep_type: p_type.clone(),
                    key: None,
                    field_decls: None,
                    contents: None,
                } // Ready to write this var decl.
            }
            RepType::HashCodeStruct(ty_string) => {
                let mut tmp = TopLevlDecl {
                    map,
                    var_name: var_name.clone(),
                    dec_type: ty_string.clone(),
                    rep_type: String::from("hashcode"),
                    key: Some(ty_string.clone()),
                    field_decls: Some(Vec::new()),
                    contents: None,
                };
                tmp.build_fields(depth_limit, &mut do_write);

                // Error checking
                if !do_write {
                    // Any "fields" are invalid, but tmp could be an enum/union and pointer is valid.
                    match &mut tmp.field_decls {
                        None => panic!("Expected some field_decls 1"),
                        Some(field_decls) => {
                            let mut j = 0;
                            while j < field_decls.len() {
                                field_decls[j].var_name = String::from("false");
                                j += 1;
                            }
                        }
                    }
                }
                if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                    // this record is also invalid
                    tmp.var_name = String::from("false");
                }
                tmp
            }
            RepType::PrimArray(p_type) => {
                TopLevlDecl {
                    map,
                    var_name: var_name.clone(),
                    dec_type: format!("{}[]", p_type),
                    rep_type: String::from("hashcode"),
                    key: None,
                    field_decls: None,
                    contents: Some(ArrayContents {
                        map,
                        var_name: format!("{}[..]", var_name),
                        dec_type: format!("{}[]", p_type),
                        rep_type: format!("{}[]", p_type),
                        enclosing_var: var_name.clone(),
                        key: None,
                        sub_contents: None,
                    }), // Ready to write this var_decl.
                }
            }
            RepType::HashCodeArray(ty_string) => {
                let mut tmp = TopLevlDecl {
                    map,
                    var_name: var_name.clone(),
                    dec_type: format!("{}[]", ty_string),
                    rep_type: String::from("hashcode"),
                    key: Some(ty_string.clone()),
                    field_decls: None,
                    contents: Some(ArrayContents {
                        map,
                        var_name: format!("{}[..]", var_name),
                        dec_type: format!("{}[]", ty_string),
                        rep_type: String::from("hashcode[]"),
                        enclosing_var: var_name.clone(),
                        key: Some(ty_string.clone()),
                        sub_contents: Some(Vec::new()),
                    }),
                };
                match &mut tmp.contents {
                    None => panic!(""),
                    Some(contents) => {
                        contents.build_contents(depth_limit - 1, &mut do_write);

                        // Error checking: note for this and similar, tmp.contents valid is equivalent to tmp valid, if we have Vec of enums, contents is pointers.
                        if !do_write {
                            // Any "fields" are invalid, but tmp could be an enum/union and pointers is valid.
                            match &mut contents.sub_contents {
                                None => panic!("Expected some field_decls 1"),
                                Some(sub_contents) => {
                                    let mut j = 0;
                                    while j < sub_contents.len() {
                                        sub_contents[j].var_name = String::from("false");
                                        j += 1;
                                    }
                                }
                            }
                        }

                        if ty_string.starts_with("Option") || ty_string.starts_with("Result") {
                            // this record is also invalid
                            tmp.var_name = String::from("false");
                        }
                    }
                }
                tmp
            }
        };
        var_decls.push(toplevl_decl);
        i += 1;
    }

    var_decls
}

impl<'a> Visitor<'a> for DaikonDeclsVisitor<'a> {
    // Process a new function and write it to the decls file.
    fn visit_fn(&mut self, fk: FnKind<'a>, _span: rustc_span::Span, _id: rustc_ast::NodeId) {
        match &fk {
            FnKind::Fn(_, _, f) => {
                if !f.ident.as_str().starts_with("dtrace") {
                    let ppt_name = String::from(f.ident.as_str());
                    write_entry(ppt_name.clone());
                    let param_to_block_idx = map_params(&f.sig.decl);
                    let mut param_decls = grok_fn_sig(&f.sig.decl, self.map, self.depth_limit);
                    let mut i = 0;
                    while i < param_decls.len() {
                        param_decls[i].write();
                        i += 1;
                    }
                    write_newline();
                    match &f.body {
                        None => {}
                        Some(body) => {
                            // By now, all exit ppts are
                            // explicit Semi(Ret) stmts.
                            self.grok_fn_body(
                                ppt_name.clone(),
                                body,
                                &mut param_decls,
                                param_to_block_idx,
                                &f.sig.decl.output,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
        visit::walk_fn(self, fk);
    }
}

// Lock on the decls file.
static DECLS: LazyLock<Mutex<Option<std::fs::File>>> = LazyLock::new(|| Mutex::new(dtrace_open()));

// Open the decls file.
fn dtrace_open() -> Option<std::fs::File> {
    let decls_path = format!("{}{}", *OUTPUT_NAME.lock().unwrap(), ".decls");
    let decls = std::path::Path::new(&decls_path);
    Some(std::fs::File::options().write(true).append(true).open(&decls).unwrap())
}

pub struct MacroExpander<'a, 'b> {
    pub cx: &'a mut ExtCtxt<'b>,
    monotonic: bool, // cf. `cx.monotonic_expander()`
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        MacroExpander { cx, monotonic }
    }

    #[allow(rustc::default_hash_types)]
    pub fn expand_crate(&mut self, krate: ast::Crate) -> ast::Crate {
        let file_path = match self.cx.source_map().span_to_filename(krate.spans.inner_span) {
            FileName::Real(name) => name
                .into_local_path()
                .expect("attempting to resolve a file path in an external file"),
            other => PathBuf::from(other.prefer_local().to_string()),
        };
        let dir_path = file_path.parent().unwrap_or(&file_path).to_owned();
        self.cx.root_path = dir_path.clone();
        self.cx.current_expansion.module = Rc::new(ModuleData {
            mod_path: vec![Ident::with_dummy_span(self.cx.ecfg.crate_name)],
            file_path_stack: vec![file_path],
            dir_path,
        });
        let krate = self.fully_expand_fragment(AstFragment::Crate(krate)).make_crate();
        // Decls pass.
        // First, pass through the entire krate building HashMap<String, Box<Item>>
        //   (value is always an ItemKind::Struct)
        // Create new decls/dtrace files. Open decls file for writing.
        // Visit the entire immutable AST with a non-mutable visitor to write the decls file,
        // skipping over functions we generated.
        if *DO_VISITOR.lock().unwrap() {
            let mut struct_map: HashMap<String, Box<Item>> = HashMap::new();
            let mut map_builder = DeclsHashMapBuilder { map: &mut struct_map };
            map_builder.visit_crate(&krate);

            // create or overwrite decls/dtrace
            let decls_path = format!("{}{}", *OUTPUT_NAME.lock().unwrap(), ".decls");
            let decls = std::path::Path::new(&decls_path);
            std::fs::File::create(&decls).unwrap();
            let dtrace_path = format!("{}{}", *OUTPUT_NAME.lock().unwrap(), ".dtrace");
            let dtrace = std::path::Path::new(&dtrace_path);
            std::fs::File::create(&dtrace).unwrap();
            write_header();
            write_newline();
            let mut decls_visitor = DaikonDeclsVisitor { map: &struct_map, depth_limit: 4 }; // off by one to match dtrace
            decls_visitor.visit_crate(&krate);
        }
        assert_eq!(krate.id, ast::CRATE_NODE_ID);
        self.cx.trace_macros_diag();
        krate
    }

    /// Recursively expand all macro invocations in this AST fragment.
    pub fn fully_expand_fragment(&mut self, input_fragment: AstFragment) -> AstFragment {
        let orig_expansion_data = self.cx.current_expansion.clone();
        let orig_force_mode = self.cx.force_mode;

        // Collect all macro invocations and replace them with placeholders.
        let (mut fragment_with_placeholders, mut invocations) =
            self.collect_invocations(input_fragment, &[]);

        // Optimization: if we resolve all imports now,
        // we'll be able to immediately resolve most of imported macros.
        self.resolve_imports();

        // Resolve paths in all invocations and produce output expanded fragments for them, but
        // do not insert them into our input AST fragment yet, only store in `expanded_fragments`.
        // The output fragments also go through expansion recursively until no invocations are left.
        // Unresolved macros produce dummy outputs as a recovery measure.
        invocations.reverse();
        let mut expanded_fragments = Vec::new();
        let mut undetermined_invocations = Vec::new();
        let (mut progress, mut force) = (false, !self.monotonic);
        loop {
            let Some((invoc, ext)) = invocations.pop() else {
                self.resolve_imports();
                if undetermined_invocations.is_empty() {
                    break;
                }
                invocations = mem::take(&mut undetermined_invocations);
                force = !progress;
                progress = false;
                if force && self.monotonic {
                    self.cx.dcx().span_delayed_bug(
                        invocations.last().unwrap().0.span(),
                        "expansion entered force mode without producing any errors",
                    );
                }
                continue;
            };

            let ext = match ext {
                Some(ext) => ext,
                None => {
                    let eager_expansion_root = if self.monotonic {
                        invoc.expansion_data.id
                    } else {
                        orig_expansion_data.id
                    };
                    match self.cx.resolver.resolve_macro_invocation(
                        &invoc,
                        eager_expansion_root,
                        force,
                    ) {
                        Ok(ext) => ext,
                        Err(Indeterminate) => {
                            // Cannot resolve, will retry this invocation later.
                            undetermined_invocations.push((invoc, None));
                            continue;
                        }
                    }
                }
            };

            let ExpansionData { depth, id: expn_id, .. } = invoc.expansion_data;
            let depth = depth - orig_expansion_data.depth;
            self.cx.current_expansion = invoc.expansion_data.clone();
            self.cx.force_mode = force;

            let fragment_kind = invoc.fragment_kind;
            match self.expand_invoc(invoc, &ext.kind) {
                ExpandResult::Ready(fragment) => {
                    let mut derive_invocations = Vec::new();
                    let derive_placeholders = self
                        .cx
                        .resolver
                        .take_derive_resolutions(expn_id)
                        .map(|derives| {
                            derive_invocations.reserve(derives.len());
                            derives
                                .into_iter()
                                .map(|DeriveResolution { path, item, exts: _, is_const }| {
                                    // FIXME: Consider using the derive resolutions (`_exts`)
                                    // instead of enqueuing the derives to be resolved again later.
                                    // Note that this can result in duplicate diagnostics.
                                    let expn_id = LocalExpnId::fresh_empty();
                                    derive_invocations.push((
                                        Invocation {
                                            kind: InvocationKind::Derive { path, item, is_const },
                                            fragment_kind,
                                            expansion_data: ExpansionData {
                                                id: expn_id,
                                                ..self.cx.current_expansion.clone()
                                            },
                                        },
                                        None,
                                    ));
                                    NodeId::placeholder_from_expn_id(expn_id)
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();

                    let (expanded_fragment, collected_invocations) =
                        self.collect_invocations(fragment, &derive_placeholders);
                    // We choose to expand any derive invocations associated with this macro
                    // invocation *before* any macro invocations collected from the output
                    // fragment.
                    derive_invocations.extend(collected_invocations);

                    progress = true;
                    if expanded_fragments.len() < depth {
                        expanded_fragments.push(Vec::new());
                    }
                    expanded_fragments[depth - 1].push((expn_id, expanded_fragment));
                    invocations.extend(derive_invocations.into_iter().rev());
                }
                ExpandResult::Retry(invoc) => {
                    if force {
                        self.cx.dcx().span_bug(
                            invoc.span(),
                            "expansion entered force mode but is still stuck",
                        );
                    } else {
                        // Cannot expand, will retry this invocation later.
                        undetermined_invocations.push((invoc, Some(ext)));
                    }
                }
            }
        }

        self.cx.current_expansion = orig_expansion_data;
        self.cx.force_mode = orig_force_mode;

        // Finally incorporate all the expanded macros into the input AST fragment.
        let mut placeholder_expander = PlaceholderExpander::default();
        while let Some(expanded_fragments) = expanded_fragments.pop() {
            for (expn_id, expanded_fragment) in expanded_fragments.into_iter().rev() {
                placeholder_expander
                    .add(NodeId::placeholder_from_expn_id(expn_id), expanded_fragment);
            }
        }
        fragment_with_placeholders.mut_visit_with(&mut placeholder_expander);
        fragment_with_placeholders
    }

    fn resolve_imports(&mut self) {
        if self.monotonic {
            self.cx.resolver.resolve_imports();
        }
    }

    /// Collects all macro invocations reachable at this time in this AST fragment, and replace
    /// them with "placeholders" - dummy macro invocations with specially crafted `NodeId`s.
    /// Then call into resolver that builds a skeleton ("reduced graph") of the fragment and
    /// prepares data for resolving paths of macro invocations.
    fn collect_invocations(
        &mut self,
        mut fragment: AstFragment,
        extra_placeholders: &[NodeId],
    ) -> (AstFragment, Vec<(Invocation, Option<Arc<SyntaxExtension>>)>) {
        // Resolve `$crate`s in the fragment for pretty-printing.
        self.cx.resolver.resolve_dollar_crates();

        let mut invocations = {
            let mut collector = InvocationCollector {
                // Non-derive macro invocations cannot see the results of cfg expansion - they
                // will either be removed along with the item, or invoked before the cfg/cfg_attr
                // attribute is expanded. Therefore, we don't need to configure the tokens
                // Derive macros *can* see the results of cfg-expansion - they are handled
                // specially in `fully_expand_fragment`
                cx: self.cx,
                invocations: Vec::new(),
                monotonic: self.monotonic,
            };
            fragment.mut_visit_with(&mut collector);
            fragment.add_placeholders(extra_placeholders);
            collector.invocations
        };

        if self.monotonic {
            self.cx
                .resolver
                .visit_ast_fragment_with_placeholders(self.cx.current_expansion.id, &fragment);

            if self.cx.sess.opts.incremental.is_some() {
                for (invoc, _) in invocations.iter_mut() {
                    let expn_id = invoc.expansion_data.id;
                    let parent_def = self.cx.resolver.invocation_parent(expn_id);
                    let span = invoc.span_mut();
                    *span = span.with_parent(Some(parent_def));
                }
            }
        }

        (fragment, invocations)
    }

    fn error_recursion_limit_reached(&mut self) -> ErrorGuaranteed {
        let expn_data = self.cx.current_expansion.id.expn_data();
        let suggested_limit = match self.cx.ecfg.recursion_limit {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };

        let guar = self.cx.dcx().emit_err(RecursionLimitReached {
            span: expn_data.call_site,
            descr: expn_data.kind.descr(),
            suggested_limit,
            crate_name: self.cx.ecfg.crate_name,
        });

        self.cx.macro_error_and_trace_macros_diag();
        guar
    }

    /// A macro's expansion does not fit in this fragment kind.
    /// For example, a non-type macro in a type position.
    fn error_wrong_fragment_kind(
        &mut self,
        kind: AstFragmentKind,
        mac: &ast::MacCall,
        span: Span,
    ) -> ErrorGuaranteed {
        let guar =
            self.cx.dcx().emit_err(WrongFragmentKind { span, kind: kind.name(), name: &mac.path });
        self.cx.macro_error_and_trace_macros_diag();
        guar
    }

    fn expand_invoc(
        &mut self,
        invoc: Invocation,
        ext: &SyntaxExtensionKind,
    ) -> ExpandResult<AstFragment, Invocation> {
        let recursion_limit = match self.cx.reduced_recursion_limit {
            Some((limit, _)) => limit,
            None => self.cx.ecfg.recursion_limit,
        };

        if !recursion_limit.value_within_limit(self.cx.current_expansion.depth) {
            let guar = match self.cx.reduced_recursion_limit {
                Some((_, guar)) => guar,
                None => self.error_recursion_limit_reached(),
            };

            // Reduce the recursion limit by half each time it triggers.
            self.cx.reduced_recursion_limit = Some((recursion_limit / 2, guar));

            return ExpandResult::Ready(invoc.fragment_kind.dummy(invoc.span(), guar));
        }

        let macro_stats = self.cx.sess.opts.unstable_opts.macro_stats;

        let (fragment_kind, span) = (invoc.fragment_kind, invoc.span());
        ExpandResult::Ready(match invoc.kind {
            InvocationKind::Bang { mac, span } => {
                if let SyntaxExtensionKind::Bang(expander) = ext {
                    match expander.expand(self.cx, span, mac.args.tokens.clone()) {
                        Ok(tok_result) => {
                            let fragment =
                                self.parse_ast_fragment(tok_result, fragment_kind, &mac.path, span);
                            if macro_stats {
                                update_bang_macro_stats(
                                    self.cx,
                                    fragment_kind,
                                    span,
                                    mac,
                                    &fragment,
                                );
                            }
                            fragment
                        }
                        Err(guar) => return ExpandResult::Ready(fragment_kind.dummy(span, guar)),
                    }
                } else if let Some(expander) = ext.as_legacy_bang() {
                    let tok_result = match expander.expand(self.cx, span, mac.args.tokens.clone()) {
                        ExpandResult::Ready(tok_result) => tok_result,
                        ExpandResult::Retry(_) => {
                            // retry the original
                            return ExpandResult::Retry(Invocation {
                                kind: InvocationKind::Bang { mac, span },
                                ..invoc
                            });
                        }
                    };
                    if let Some(fragment) = fragment_kind.make_from(tok_result) {
                        if macro_stats {
                            update_bang_macro_stats(self.cx, fragment_kind, span, mac, &fragment);
                        }
                        fragment
                    } else {
                        let guar = self.error_wrong_fragment_kind(fragment_kind, &mac, span);
                        fragment_kind.dummy(span, guar)
                    }
                } else {
                    unreachable!();
                }
            }
            InvocationKind::Attr { attr, pos, mut item, derives } => {
                if let Some(expander) = ext.as_attr() {
                    self.gate_proc_macro_input(&item);
                    self.gate_proc_macro_attr_item(span, &item);
                    let tokens = match &item {
                        // FIXME: Collect tokens and use them instead of generating
                        // fake ones. These are unstable, so it needs to be
                        // fixed prior to stabilization
                        // Fake tokens when we are invoking an inner attribute, and
                        // we are invoking it on an out-of-line module or crate.
                        Annotatable::Crate(krate) => {
                            rustc_parse::fake_token_stream_for_crate(&self.cx.sess.psess, krate)
                        }
                        Annotatable::Item(item_inner)
                            if matches!(attr.style, AttrStyle::Inner)
                                && matches!(
                                    item_inner.kind,
                                    ItemKind::Mod(
                                        _,
                                        _,
                                        ModKind::Unloaded
                                            | ModKind::Loaded(_, Inline::No { .. }, _),
                                    )
                                ) =>
                        {
                            rustc_parse::fake_token_stream_for_item(&self.cx.sess.psess, item_inner)
                        }
                        _ => item.to_tokens(),
                    };
                    let attr_item = attr.get_normal_item();
                    let safety = attr_item.unsafety;
                    if let AttrArgs::Eq { .. } = attr_item.args {
                        self.cx.dcx().emit_err(UnsupportedKeyValue { span });
                    }
                    let inner_tokens = attr_item.args.inner_tokens();
                    match expander.expand_with_safety(self.cx, safety, span, inner_tokens, tokens) {
                        Ok(tok_result) => {
                            let fragment = self.parse_ast_fragment(
                                tok_result,
                                fragment_kind,
                                &attr_item.path,
                                span,
                            );
                            if macro_stats {
                                update_attr_macro_stats(
                                    self.cx,
                                    fragment_kind,
                                    span,
                                    &attr_item.path,
                                    &attr,
                                    item,
                                    &fragment,
                                );
                            }
                            fragment
                        }
                        Err(guar) => return ExpandResult::Ready(fragment_kind.dummy(span, guar)),
                    }
                } else if let SyntaxExtensionKind::LegacyAttr(expander) = ext {
                    // `LegacyAttr` is only used for builtin attribute macros, which have their
                    // safety checked by `check_builtin_meta_item`, so we don't need to check
                    // `unsafety` here.
                    match validate_attr::parse_meta(&self.cx.sess.psess, &attr) {
                        Ok(meta) => {
                            let item_clone = macro_stats.then(|| item.clone());
                            let items = match expander.expand(self.cx, span, &meta, item, false) {
                                ExpandResult::Ready(items) => items,
                                ExpandResult::Retry(item) => {
                                    // Reassemble the original invocation for retrying.
                                    return ExpandResult::Retry(Invocation {
                                        kind: InvocationKind::Attr { attr, pos, item, derives },
                                        ..invoc
                                    });
                                }
                            };
                            if matches!(
                                fragment_kind,
                                AstFragmentKind::Expr | AstFragmentKind::MethodReceiverExpr
                            ) && items.is_empty()
                            {
                                let guar = self.cx.dcx().emit_err(RemoveExprNotSupported { span });
                                fragment_kind.dummy(span, guar)
                            } else {
                                let fragment = fragment_kind.expect_from_annotatables(items);
                                if macro_stats {
                                    update_attr_macro_stats(
                                        self.cx,
                                        fragment_kind,
                                        span,
                                        &meta.path,
                                        &attr,
                                        item_clone.unwrap(),
                                        &fragment,
                                    );
                                }
                                fragment
                            }
                        }
                        Err(err) => {
                            let _guar = err.emit();
                            fragment_kind.expect_from_annotatables(iter::once(item))
                        }
                    }
                } else if let SyntaxExtensionKind::NonMacroAttr = ext {
                    if let ast::Safety::Unsafe(span) = attr.get_normal_item().unsafety {
                        self.cx.dcx().span_err(span, "unnecessary `unsafe` on safe attribute");
                    }
                    // `-Zmacro-stats` ignores these because they don't do any real expansion.
                    self.cx.expanded_inert_attrs.mark(&attr);
                    item.visit_attrs(|attrs| attrs.insert(pos, attr));
                    fragment_kind.expect_from_annotatables(iter::once(item))
                } else {
                    unreachable!();
                }
            }
            InvocationKind::Derive { path, item, is_const } => match ext {
                SyntaxExtensionKind::Derive(expander)
                | SyntaxExtensionKind::LegacyDerive(expander) => {
                    if let SyntaxExtensionKind::Derive(..) = ext {
                        self.gate_proc_macro_input(&item);
                    }
                    // The `MetaItem` representing the trait to derive can't
                    // have an unsafe around it (as of now).
                    let meta = ast::MetaItem {
                        unsafety: ast::Safety::Default,
                        kind: MetaItemKind::Word,
                        span,
                        path,
                    };
                    let items = match expander.expand(self.cx, span, &meta, item, is_const) {
                        ExpandResult::Ready(items) => items,
                        ExpandResult::Retry(item) => {
                            // Reassemble the original invocation for retrying.
                            return ExpandResult::Retry(Invocation {
                                kind: InvocationKind::Derive { path: meta.path, item, is_const },
                                ..invoc
                            });
                        }
                    };
                    let fragment = fragment_kind.expect_from_annotatables(items);
                    if macro_stats {
                        update_derive_macro_stats(
                            self.cx,
                            fragment_kind,
                            span,
                            &meta.path,
                            &fragment,
                        );
                    }
                    fragment
                }
                SyntaxExtensionKind::MacroRules(expander)
                    if expander.kinds().contains(MacroKinds::DERIVE) =>
                {
                    if is_const {
                        let guar = self
                            .cx
                            .dcx()
                            .span_err(span, "macro `derive` does not support const derives");
                        return ExpandResult::Ready(fragment_kind.dummy(span, guar));
                    }
                    let body = item.to_tokens();
                    match expander.expand_derive(self.cx, span, &body) {
                        Ok(tok_result) => {
                            let fragment =
                                self.parse_ast_fragment(tok_result, fragment_kind, &path, span);
                            if macro_stats {
                                update_derive_macro_stats(
                                    self.cx,
                                    fragment_kind,
                                    span,
                                    &path,
                                    &fragment,
                                );
                            }
                            fragment
                        }
                        Err(guar) => return ExpandResult::Ready(fragment_kind.dummy(span, guar)),
                    }
                }
                _ => unreachable!(),
            },
            InvocationKind::GlobDelegation { item, of_trait } => {
                let AssocItemKind::DelegationMac(deleg) = &item.kind else { unreachable!() };
                let suffixes = match ext {
                    SyntaxExtensionKind::GlobDelegation(expander) => match expander.expand(self.cx)
                    {
                        ExpandResult::Ready(suffixes) => suffixes,
                        ExpandResult::Retry(()) => {
                            // Reassemble the original invocation for retrying.
                            return ExpandResult::Retry(Invocation {
                                kind: InvocationKind::GlobDelegation { item, of_trait },
                                ..invoc
                            });
                        }
                    },
                    SyntaxExtensionKind::Bang(..) => {
                        let msg = "expanded a dummy glob delegation";
                        let guar = self.cx.dcx().span_delayed_bug(span, msg);
                        return ExpandResult::Ready(fragment_kind.dummy(span, guar));
                    }
                    _ => unreachable!(),
                };

                type Node = AstNodeWrapper<Box<ast::AssocItem>, ImplItemTag>;
                let single_delegations = build_single_delegations::<Node>(
                    self.cx, deleg, &item, &suffixes, item.span, true,
                );
                // `-Zmacro-stats` ignores these because they don't seem important.
                fragment_kind.expect_from_annotatables(single_delegations.map(|item| {
                    Annotatable::AssocItem(Box::new(item), AssocCtxt::Impl { of_trait })
                }))
            }
        })
    }

    #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
    fn gate_proc_macro_attr_item(&self, span: Span, item: &Annotatable) {
        let kind = match item {
            Annotatable::Item(_)
            | Annotatable::AssocItem(..)
            | Annotatable::ForeignItem(_)
            | Annotatable::Crate(..) => return,
            Annotatable::Stmt(stmt) => {
                // Attributes are stable on item statements,
                // but unstable on all other kinds of statements
                if stmt.is_item() {
                    return;
                }
                "statements"
            }
            Annotatable::Expr(_) => "expressions",
            Annotatable::Arm(..)
            | Annotatable::ExprField(..)
            | Annotatable::PatField(..)
            | Annotatable::GenericParam(..)
            | Annotatable::Param(..)
            | Annotatable::FieldDef(..)
            | Annotatable::Variant(..)
            | Annotatable::WherePredicate(..) => panic!("unexpected annotatable"),
        };
        if self.cx.ecfg.features.proc_macro_hygiene() {
            return;
        }
        feature_err(
            &self.cx.sess,
            sym::proc_macro_hygiene,
            span,
            format!("custom attributes cannot be applied to {kind}"),
        )
        .emit();
    }

    fn gate_proc_macro_input(&self, annotatable: &Annotatable) {
        struct GateProcMacroInput<'a> {
            sess: &'a Session,
        }

        impl<'ast, 'a> Visitor<'ast> for GateProcMacroInput<'a> {
            fn visit_item(&mut self, item: &'ast ast::Item) {
                match &item.kind {
                    ItemKind::Mod(_, _, mod_kind)
                        if !matches!(mod_kind, ModKind::Loaded(_, Inline::Yes, _)) =>
                    {
                        feature_err(
                            self.sess,
                            sym::proc_macro_hygiene,
                            item.span,
                            fluent_generated::expand_non_inline_modules_in_proc_macro_input_are_unstable,
                        )
                        .emit();
                    }
                    _ => {}
                }

                visit::walk_item(self, item);
            }
        }

        if !self.cx.ecfg.features.proc_macro_hygiene() {
            annotatable.visit_with(&mut GateProcMacroInput { sess: &self.cx.sess });
        }
    }

    fn parse_ast_fragment(
        &mut self,
        toks: TokenStream,
        kind: AstFragmentKind,
        path: &ast::Path,
        span: Span,
    ) -> AstFragment {
        let mut parser = self.cx.new_parser_from_tts(toks);
        match parse_ast_fragment(&mut parser, kind) {
            Ok(fragment) => {
                ensure_complete_parse(&parser, path, kind.name(), span);
                fragment
            }
            Err(mut err) => {
                if err.span.is_dummy() {
                    err.span(span);
                }
                annotate_err_with_kind(&mut err, kind, span);
                let guar = err.emit();
                self.cx.macro_error_and_trace_macros_diag();
                kind.dummy(span, guar)
            }
        }
    }
}

pub fn parse_ast_fragment<'a>(
    this: &mut Parser<'a>,
    kind: AstFragmentKind,
) -> PResult<'a, AstFragment> {
    Ok(match kind {
        AstFragmentKind::Items => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_item(ForceCollect::No)? {
                items.push(item);
            }
            AstFragment::Items(items)
        }
        AstFragmentKind::TraitItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_trait_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::TraitItems(items)
        }
        AstFragmentKind::ImplItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_impl_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::ImplItems(items)
        }
        AstFragmentKind::TraitImplItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_impl_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::TraitImplItems(items)
        }
        AstFragmentKind::ForeignItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_foreign_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::ForeignItems(items)
        }
        AstFragmentKind::Stmts => {
            let mut stmts = SmallVec::new();
            // Won't make progress on a `}`.
            while this.token != token::Eof && this.token != token::CloseBrace {
                if let Some(stmt) = this.parse_full_stmt(AttemptLocalParseRecovery::Yes)? {
                    stmts.push(stmt);
                }
            }
            AstFragment::Stmts(stmts)
        }
        AstFragmentKind::Expr => AstFragment::Expr(this.parse_expr()?),
        AstFragmentKind::MethodReceiverExpr => AstFragment::MethodReceiverExpr(this.parse_expr()?),
        AstFragmentKind::OptExpr => {
            if this.token != token::Eof {
                AstFragment::OptExpr(Some(this.parse_expr()?))
            } else {
                AstFragment::OptExpr(None)
            }
        }
        AstFragmentKind::Ty => AstFragment::Ty(this.parse_ty()?),
        AstFragmentKind::Pat => AstFragment::Pat(this.parse_pat_allow_top_guard(
            None,
            RecoverComma::No,
            RecoverColon::Yes,
            CommaRecoveryMode::LikelyTuple,
        )?),
        AstFragmentKind::Crate => AstFragment::Crate(this.parse_crate_mod()?),
        AstFragmentKind::Arms
        | AstFragmentKind::ExprFields
        | AstFragmentKind::PatFields
        | AstFragmentKind::GenericParams
        | AstFragmentKind::Params
        | AstFragmentKind::FieldDefs
        | AstFragmentKind::Variants
        | AstFragmentKind::WherePredicates => panic!("unexpected AST fragment kind"),
    })
}

pub(crate) fn ensure_complete_parse<'a>(
    parser: &Parser<'a>,
    macro_path: &ast::Path,
    kind_name: &str,
    span: Span,
) {
    if parser.token != token::Eof {
        let descr = token_descr(&parser.token);
        // Avoid emitting backtrace info twice.
        let def_site_span = parser.token.span.with_ctxt(SyntaxContext::root());

        let semi_span = parser.psess.source_map().next_point(span);
        let add_semicolon = match &parser.psess.source_map().span_to_snippet(semi_span) {
            Ok(snippet) if &snippet[..] != ";" && kind_name == "expression" => {
                Some(span.shrink_to_hi())
            }
            _ => None,
        };

        let expands_to_match_arm = kind_name == "pattern" && parser.token == token::FatArrow;

        parser.dcx().emit_err(IncompleteParse {
            span: def_site_span,
            descr,
            label_span: span,
            macro_path,
            kind_name,
            expands_to_match_arm,
            add_semicolon,
        });
    }
}

/// Wraps a call to `walk_*` / `walk_flat_map_*`
/// for an AST node that supports attributes
/// (see the `Annotatable` enum)
/// This method assigns a `NodeId`, and sets that `NodeId`
/// as our current 'lint node id'. If a macro call is found
/// inside this AST node, we will use this AST node's `NodeId`
/// to emit lints associated with that macro (allowing
/// `#[allow]` / `#[deny]` to be applied close to
/// the macro invocation).
///
/// Do *not* call this for a macro AST node
/// (e.g. `ExprKind::MacCall`) - we cannot emit lints
/// at these AST nodes, since they are removed and
/// replaced with the result of macro expansion.
///
/// All other `NodeId`s are assigned by `visit_id`.
/// * `self` is the 'self' parameter for the current method,
/// * `id` is a mutable reference to the `NodeId` field
///    of the current AST node.
/// * `closure` is a closure that executes the
///   `walk_*` / `walk_flat_map_*` method
///   for the current AST node.
macro_rules! assign_id {
    ($self:ident, $id:expr, $closure:expr) => {{
        let old_id = $self.cx.current_expansion.lint_node_id;
        if $self.monotonic {
            debug_assert_eq!(*$id, ast::DUMMY_NODE_ID);
            let new_id = $self.cx.resolver.next_node_id();
            *$id = new_id;
            $self.cx.current_expansion.lint_node_id = new_id;
        }
        let ret = ($closure)();
        $self.cx.current_expansion.lint_node_id = old_id;
        ret
    }};
}

enum AddSemicolon {
    Yes,
    No,
}

/// A trait implemented for all `AstFragment` nodes and providing all pieces
/// of functionality used by `InvocationCollector`.
trait InvocationCollectorNode: HasAttrs + HasNodeId + Sized {
    type OutputTy = SmallVec<[Self; 1]>;
    type ItemKind = ItemKind;
    const KIND: AstFragmentKind;
    fn to_annotatable(self) -> Annotatable;
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy;
    fn descr() -> &'static str {
        unreachable!()
    }
    fn walk_flat_map(self, _collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        unreachable!()
    }
    fn walk(&mut self, _collector: &mut InvocationCollector<'_, '_>) {
        unreachable!()
    }
    fn is_mac_call(&self) -> bool {
        false
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        unreachable!()
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        None
    }
    fn delegation_item_kind(_deleg: Box<ast::Delegation>) -> Self::ItemKind {
        unreachable!()
    }
    fn from_item(_item: ast::Item<Self::ItemKind>) -> Self {
        unreachable!()
    }
    fn flatten_outputs(_outputs: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        unreachable!()
    }
    fn pre_flat_map_node_collect_attr(_cfg: &StripUnconfigured<'_>, _attr: &ast::Attribute) {}
    fn post_flat_map_node_collect_bang(_output: &mut Self::OutputTy, _add_semicolon: AddSemicolon) {
    }
    fn wrap_flat_map_node_walk_flat_map(
        node: Self,
        collector: &mut InvocationCollector<'_, '_>,
        walk_flat_map: impl FnOnce(Self, &mut InvocationCollector<'_, '_>) -> Self::OutputTy,
    ) -> Result<Self::OutputTy, Self> {
        Ok(walk_flat_map(node, collector))
    }
    fn expand_cfg_false(
        &mut self,
        collector: &mut InvocationCollector<'_, '_>,
        _pos: usize,
        span: Span,
    ) {
        collector.cx.dcx().emit_err(RemoveNodeNotSupported { span, descr: Self::descr() });
    }

    /// All of the identifiers (items) declared by this node.
    /// This is an approximation and should only be used for diagnostics.
    fn declared_idents(&self) -> Vec<Ident> {
        vec![]
    }
}

impl InvocationCollectorNode for Box<ast::Item> {
    const KIND: AstFragmentKind = AstFragmentKind::Items;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Item(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_items()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_item(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        match self.kind {
            ItemKind::MacCall(mac) => (mac, self.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        match &self.kind {
            ItemKind::DelegationMac(deleg) => Some((deleg, self)),
            _ => None,
        }
    }
    fn delegation_item_kind(deleg: Box<ast::Delegation>) -> Self::ItemKind {
        ItemKind::Delegation(deleg)
    }
    fn from_item(item: ast::Item<Self::ItemKind>) -> Self {
        Box::new(item)
    }
    fn flatten_outputs(items: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        items.flatten().collect()
    }
    fn wrap_flat_map_node_walk_flat_map(
        mut node: Self,
        collector: &mut InvocationCollector<'_, '_>,
        walk_flat_map: impl FnOnce(Self, &mut InvocationCollector<'_, '_>) -> Self::OutputTy,
    ) -> Result<Self::OutputTy, Self> {
        if !matches!(node.kind, ItemKind::Mod(..)) {
            return Ok(walk_flat_map(node, collector));
        }

        // Work around borrow checker not seeing through `P`'s deref.
        let (span, mut attrs) = (node.span, mem::take(&mut node.attrs));
        let ItemKind::Mod(_, ident, ref mut mod_kind) = node.kind else { unreachable!() };
        let ecx = &mut collector.cx;
        let (file_path, dir_path, dir_ownership) = match mod_kind {
            ModKind::Loaded(_, inline, _) => {
                // Inline `mod foo { ... }`, but we still need to push directories.
                let (dir_path, dir_ownership) = mod_dir_path(
                    ecx.sess,
                    ident,
                    &attrs,
                    &ecx.current_expansion.module,
                    ecx.current_expansion.dir_ownership,
                    *inline,
                );
                // If the module was parsed from an external file, recover its path.
                // This lets `parse_external_mod` catch cycles if it's self-referential.
                let file_path = match inline {
                    Inline::Yes => None,
                    Inline::No { .. } => mod_file_path_from_attr(ecx.sess, &attrs, &dir_path),
                };
                node.attrs = attrs;
                (file_path, dir_path, dir_ownership)
            }
            ModKind::Unloaded => {
                // We have an outline `mod foo;` so we need to parse the file.
                let old_attrs_len = attrs.len();
                let ParsedExternalMod {
                    items,
                    spans,
                    file_path,
                    dir_path,
                    dir_ownership,
                    had_parse_error,
                } = parse_external_mod(
                    ecx.sess,
                    ident,
                    span,
                    &ecx.current_expansion.module,
                    ecx.current_expansion.dir_ownership,
                    &mut attrs,
                );

                if let Some(lint_store) = ecx.lint_store {
                    lint_store.pre_expansion_lint(
                        ecx.sess,
                        ecx.ecfg.features,
                        ecx.resolver.registered_tools(),
                        ecx.current_expansion.lint_node_id,
                        &attrs,
                        &items,
                        ident.name,
                    );
                }

                *mod_kind = ModKind::Loaded(items, Inline::No { had_parse_error }, spans);
                node.attrs = attrs;
                if node.attrs.len() > old_attrs_len {
                    // If we loaded an out-of-line module and added some inner attributes,
                    // then we need to re-configure it and re-collect attributes for
                    // resolution and expansion.
                    return Err(node);
                }
                (Some(file_path), dir_path, dir_ownership)
            }
        };

        // Set the module info before we flat map.
        let mut module = ecx.current_expansion.module.with_dir_path(dir_path);
        module.mod_path.push(ident);
        if let Some(file_path) = file_path {
            module.file_path_stack.push(file_path);
        }

        let orig_module = mem::replace(&mut ecx.current_expansion.module, Rc::new(module));
        let orig_dir_ownership =
            mem::replace(&mut ecx.current_expansion.dir_ownership, dir_ownership);

        let res = Ok(walk_flat_map(node, collector));

        collector.cx.current_expansion.dir_ownership = orig_dir_ownership;
        collector.cx.current_expansion.module = orig_module;
        res
    }

    fn declared_idents(&self) -> Vec<Ident> {
        if let ItemKind::Use(ut) = &self.kind {
            fn collect_use_tree_leaves(ut: &ast::UseTree, idents: &mut Vec<Ident>) {
                match &ut.kind {
                    ast::UseTreeKind::Glob => {}
                    ast::UseTreeKind::Simple(_) => idents.push(ut.ident()),
                    ast::UseTreeKind::Nested { items, .. } => {
                        for (ut, _) in items {
                            collect_use_tree_leaves(ut, idents);
                        }
                    }
                }
            }
            let mut idents = Vec::new();
            collect_use_tree_leaves(&ut, &mut idents);
            idents
        } else {
            self.kind.ident().into_iter().collect()
        }
    }
}

struct TraitItemTag;
impl InvocationCollectorNode for AstNodeWrapper<Box<ast::AssocItem>, TraitItemTag> {
    type OutputTy = SmallVec<[Box<ast::AssocItem>; 1]>;
    type ItemKind = AssocItemKind;
    const KIND: AstFragmentKind = AstFragmentKind::TraitItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::AssocItem(self.wrapped, AssocCtxt::Trait)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_trait_items()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_assoc_item(collector, self.wrapped, AssocCtxt::Trait)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, AssocItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        let item = self.wrapped;
        match item.kind {
            AssocItemKind::MacCall(mac) => (mac, item.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        match &self.wrapped.kind {
            AssocItemKind::DelegationMac(deleg) => Some((deleg, &self.wrapped)),
            _ => None,
        }
    }
    fn delegation_item_kind(deleg: Box<ast::Delegation>) -> Self::ItemKind {
        AssocItemKind::Delegation(deleg)
    }
    fn from_item(item: ast::Item<Self::ItemKind>) -> Self {
        AstNodeWrapper::new(Box::new(item), TraitItemTag)
    }
    fn flatten_outputs(items: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        items.flatten().collect()
    }
}

struct ImplItemTag;
impl InvocationCollectorNode for AstNodeWrapper<Box<ast::AssocItem>, ImplItemTag> {
    type OutputTy = SmallVec<[Box<ast::AssocItem>; 1]>;
    type ItemKind = AssocItemKind;
    const KIND: AstFragmentKind = AstFragmentKind::ImplItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::AssocItem(self.wrapped, AssocCtxt::Impl { of_trait: false })
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_impl_items()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_assoc_item(collector, self.wrapped, AssocCtxt::Impl { of_trait: false })
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, AssocItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        let item = self.wrapped;
        match item.kind {
            AssocItemKind::MacCall(mac) => (mac, item.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        match &self.wrapped.kind {
            AssocItemKind::DelegationMac(deleg) => Some((deleg, &self.wrapped)),
            _ => None,
        }
    }
    fn delegation_item_kind(deleg: Box<ast::Delegation>) -> Self::ItemKind {
        AssocItemKind::Delegation(deleg)
    }
    fn from_item(item: ast::Item<Self::ItemKind>) -> Self {
        AstNodeWrapper::new(Box::new(item), ImplItemTag)
    }
    fn flatten_outputs(items: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        items.flatten().collect()
    }
}

struct TraitImplItemTag;
impl InvocationCollectorNode for AstNodeWrapper<Box<ast::AssocItem>, TraitImplItemTag> {
    type OutputTy = SmallVec<[Box<ast::AssocItem>; 1]>;
    type ItemKind = AssocItemKind;
    const KIND: AstFragmentKind = AstFragmentKind::TraitImplItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::AssocItem(self.wrapped, AssocCtxt::Impl { of_trait: true })
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_trait_impl_items()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_assoc_item(collector, self.wrapped, AssocCtxt::Impl { of_trait: true })
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, AssocItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        let item = self.wrapped;
        match item.kind {
            AssocItemKind::MacCall(mac) => (mac, item.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        match &self.wrapped.kind {
            AssocItemKind::DelegationMac(deleg) => Some((deleg, &self.wrapped)),
            _ => None,
        }
    }
    fn delegation_item_kind(deleg: Box<ast::Delegation>) -> Self::ItemKind {
        AssocItemKind::Delegation(deleg)
    }
    fn from_item(item: ast::Item<Self::ItemKind>) -> Self {
        AstNodeWrapper::new(Box::new(item), TraitImplItemTag)
    }
    fn flatten_outputs(items: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        items.flatten().collect()
    }
}

impl InvocationCollectorNode for Box<ast::ForeignItem> {
    const KIND: AstFragmentKind = AstFragmentKind::ForeignItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::ForeignItem(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_foreign_items()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_foreign_item(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ForeignItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        match self.kind {
            ForeignItemKind::MacCall(mac) => (mac, self.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for ast::Variant {
    const KIND: AstFragmentKind = AstFragmentKind::Variants;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Variant(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_variants()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_variant(collector, self)
    }
}

impl InvocationCollectorNode for ast::WherePredicate {
    const KIND: AstFragmentKind = AstFragmentKind::WherePredicates;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::WherePredicate(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_where_predicates()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_where_predicate(collector, self)
    }
}

impl InvocationCollectorNode for ast::FieldDef {
    const KIND: AstFragmentKind = AstFragmentKind::FieldDefs;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::FieldDef(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_field_defs()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_field_def(collector, self)
    }
}

impl InvocationCollectorNode for ast::PatField {
    const KIND: AstFragmentKind = AstFragmentKind::PatFields;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::PatField(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_pat_fields()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_pat_field(collector, self)
    }
}

impl InvocationCollectorNode for ast::ExprField {
    const KIND: AstFragmentKind = AstFragmentKind::ExprFields;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::ExprField(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_expr_fields()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_expr_field(collector, self)
    }
}

impl InvocationCollectorNode for ast::Param {
    const KIND: AstFragmentKind = AstFragmentKind::Params;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Param(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_params()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_param(collector, self)
    }
}

impl InvocationCollectorNode for ast::GenericParam {
    const KIND: AstFragmentKind = AstFragmentKind::GenericParams;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::GenericParam(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_generic_params()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_generic_param(collector, self)
    }
}

impl InvocationCollectorNode for ast::Arm {
    const KIND: AstFragmentKind = AstFragmentKind::Arms;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Arm(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_arms()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_arm(collector, self)
    }
}

impl InvocationCollectorNode for ast::Stmt {
    const KIND: AstFragmentKind = AstFragmentKind::Stmts;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Stmt(Box::new(self))
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_stmts()
    }
    fn walk_flat_map(self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_flat_map_stmt(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        match &self.kind {
            StmtKind::MacCall(..) => true,
            StmtKind::Item(item) => matches!(item.kind, ItemKind::MacCall(..)),
            StmtKind::Semi(expr) => matches!(expr.kind, ExprKind::MacCall(..)),
            StmtKind::Expr(..) => unreachable!(),
            StmtKind::Let(..) | StmtKind::Empty => false,
        }
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        // We pull macro invocations (both attributes and fn-like macro calls) out of their
        // `StmtKind`s and treat them as statement macro invocations, not as items or expressions.
        let (add_semicolon, mac, attrs) = match self.kind {
            StmtKind::MacCall(mac) => {
                let ast::MacCallStmt { mac, style, attrs, .. } = *mac;
                (style == MacStmtStyle::Semicolon, mac, attrs)
            }
            StmtKind::Item(item) => match *item {
                ast::Item { kind: ItemKind::MacCall(mac), attrs, .. } => {
                    (mac.args.need_semicolon(), mac, attrs)
                }
                _ => unreachable!(),
            },
            StmtKind::Semi(expr) => match *expr {
                ast::Expr { kind: ExprKind::MacCall(mac), attrs, .. } => {
                    (mac.args.need_semicolon(), mac, attrs)
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        (mac, attrs, if add_semicolon { AddSemicolon::Yes } else { AddSemicolon::No })
    }
    fn delegation(&self) -> Option<(&ast::DelegationMac, &ast::Item<Self::ItemKind>)> {
        match &self.kind {
            StmtKind::Item(item) => match &item.kind {
                ItemKind::DelegationMac(deleg) => Some((deleg, item)),
                _ => None,
            },
            _ => None,
        }
    }
    fn delegation_item_kind(deleg: Box<ast::Delegation>) -> Self::ItemKind {
        ItemKind::Delegation(deleg)
    }
    fn from_item(item: ast::Item<Self::ItemKind>) -> Self {
        ast::Stmt { id: ast::DUMMY_NODE_ID, span: item.span, kind: StmtKind::Item(Box::new(item)) }
    }
    fn flatten_outputs(items: impl Iterator<Item = Self::OutputTy>) -> Self::OutputTy {
        items.flatten().collect()
    }
    fn post_flat_map_node_collect_bang(stmts: &mut Self::OutputTy, add_semicolon: AddSemicolon) {
        // If this is a macro invocation with a semicolon, then apply that
        // semicolon to the final statement produced by expansion.
        if matches!(add_semicolon, AddSemicolon::Yes) {
            if let Some(stmt) = stmts.pop() {
                stmts.push(stmt.add_trailing_semicolon());
            }
        }
    }
}

impl InvocationCollectorNode for ast::Crate {
    type OutputTy = ast::Crate;
    const KIND: AstFragmentKind = AstFragmentKind::Crate;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Crate(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_crate()
    }
    fn walk(&mut self, collector: &mut InvocationCollector<'_, '_>) {
        walk_crate(collector, self)
    }
    fn expand_cfg_false(
        &mut self,
        collector: &mut InvocationCollector<'_, '_>,
        pos: usize,
        _span: Span,
    ) {
        // Attributes above `cfg(FALSE)` are left in place, because we may want to configure
        // some global crate properties even on fully unconfigured crates.
        self.attrs.truncate(pos);
        // Standard prelude imports are left in the crate for backward compatibility.
        self.items.truncate(collector.cx.num_standard_library_imports);
    }
}

impl InvocationCollectorNode for ast::Ty {
    type OutputTy = Box<ast::Ty>;
    const KIND: AstFragmentKind = AstFragmentKind::Ty;
    fn to_annotatable(self) -> Annotatable {
        unreachable!()
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_ty()
    }
    fn walk(&mut self, collector: &mut InvocationCollector<'_, '_>) {
        // Save the pre-expanded name of this `ImplTrait`, so that later when defining
        // an APIT we use a name that doesn't have any placeholder fragments in it.
        if let ast::TyKind::ImplTrait(..) = self.kind {
            // HACK: pprust breaks strings with newlines when the type
            // gets too long. We don't want these to show up in compiler
            // output or built artifacts, so replace them here...
            // Perhaps we should instead format APITs more robustly.
            let name = Symbol::intern(&pprust::ty_to_string(self).replace('\n', " "));
            collector.cx.resolver.insert_impl_trait_name(self.id, name);
        }
        walk_ty(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ast::TyKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        match self.kind {
            TyKind::MacCall(mac) => (mac, AttrVec::new(), AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for ast::Pat {
    type OutputTy = Box<ast::Pat>;
    const KIND: AstFragmentKind = AstFragmentKind::Pat;
    fn to_annotatable(self) -> Annotatable {
        unreachable!()
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_pat()
    }
    fn walk(&mut self, collector: &mut InvocationCollector<'_, '_>) {
        walk_pat(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, PatKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        match self.kind {
            PatKind::MacCall(mac) => (mac, AttrVec::new(), AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for ast::Expr {
    type OutputTy = Box<ast::Expr>;
    const KIND: AstFragmentKind = AstFragmentKind::Expr;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(Box::new(self))
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_expr()
    }
    fn descr() -> &'static str {
        "an expression"
    }
    fn walk(&mut self, collector: &mut InvocationCollector<'_, '_>) {
        walk_expr(collector, self)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        match self.kind {
            ExprKind::MacCall(mac) => (mac, self.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

struct OptExprTag;
impl InvocationCollectorNode for AstNodeWrapper<Box<ast::Expr>, OptExprTag> {
    type OutputTy = Option<Box<ast::Expr>>;
    const KIND: AstFragmentKind = AstFragmentKind::OptExpr;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(self.wrapped)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_opt_expr()
    }
    fn walk_flat_map(mut self, collector: &mut InvocationCollector<'_, '_>) -> Self::OutputTy {
        walk_expr(collector, &mut self.wrapped);
        Some(self.wrapped)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, ast::ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        let node = self.wrapped;
        match node.kind {
            ExprKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn pre_flat_map_node_collect_attr(cfg: &StripUnconfigured<'_>, attr: &ast::Attribute) {
        cfg.maybe_emit_expr_attr_err(attr);
    }
}

/// This struct is a hack to workaround unstable of `stmt_expr_attributes`.
/// It can be removed once that feature is stabilized.
struct MethodReceiverTag;

impl InvocationCollectorNode for AstNodeWrapper<ast::Expr, MethodReceiverTag> {
    type OutputTy = AstNodeWrapper<Box<ast::Expr>, MethodReceiverTag>;
    const KIND: AstFragmentKind = AstFragmentKind::MethodReceiverExpr;
    fn descr() -> &'static str {
        "an expression"
    }
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(Box::new(self.wrapped))
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        AstNodeWrapper::new(fragment.make_method_receiver_expr(), MethodReceiverTag)
    }
    fn walk(&mut self, collector: &mut InvocationCollector<'_, '_>) {
        walk_expr(collector, &mut self.wrapped)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, ast::ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (Box<ast::MacCall>, ast::AttrVec, AddSemicolon) {
        let node = self.wrapped;
        match node.kind {
            ExprKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

fn build_single_delegations<'a, Node: InvocationCollectorNode>(
    ecx: &ExtCtxt<'_>,
    deleg: &'a ast::DelegationMac,
    item: &'a ast::Item<Node::ItemKind>,
    suffixes: &'a [(Ident, Option<Ident>)],
    item_span: Span,
    from_glob: bool,
) -> impl Iterator<Item = ast::Item<Node::ItemKind>> + 'a {
    if suffixes.is_empty() {
        // Report an error for now, to avoid keeping stem for resolution and
        // stability checks.
        let kind = String::from(if from_glob { "glob" } else { "list" });
        ecx.dcx().emit_err(EmptyDelegationMac { span: item.span, kind });
    }

    suffixes.iter().map(move |&(ident, rename)| {
        let mut path = deleg.prefix.clone();
        path.segments.push(ast::PathSegment { ident, id: ast::DUMMY_NODE_ID, args: None });

        ast::Item {
            attrs: item.attrs.clone(),
            id: ast::DUMMY_NODE_ID,
            span: if from_glob { item_span } else { ident.span },
            vis: item.vis.clone(),
            kind: Node::delegation_item_kind(Box::new(ast::Delegation {
                id: ast::DUMMY_NODE_ID,
                qself: deleg.qself.clone(),
                path,
                ident: rename.unwrap_or(ident),
                rename,
                body: deleg.body.clone(),
                from_glob,
            })),
            tokens: None,
        }
    })
}

/// Required for `visit_node` obtained an owned `Node` from `&mut Node`.
trait DummyAstNode {
    fn dummy() -> Self;
}

impl DummyAstNode for ast::Crate {
    fn dummy() -> Self {
        ast::Crate {
            attrs: Default::default(),
            items: Default::default(),
            spans: Default::default(),
            id: DUMMY_NODE_ID,
            is_placeholder: Default::default(),
        }
    }
}

impl DummyAstNode for ast::Ty {
    fn dummy() -> Self {
        ast::Ty {
            id: DUMMY_NODE_ID,
            kind: TyKind::Dummy,
            span: Default::default(),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for ast::Pat {
    fn dummy() -> Self {
        ast::Pat {
            id: DUMMY_NODE_ID,
            kind: PatKind::Wild,
            span: Default::default(),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for ast::Expr {
    fn dummy() -> Self {
        ast::Expr::dummy()
    }
}

impl DummyAstNode for AstNodeWrapper<ast::Expr, MethodReceiverTag> {
    fn dummy() -> Self {
        AstNodeWrapper::new(ast::Expr::dummy(), MethodReceiverTag)
    }
}

struct InvocationCollector<'a, 'b> {
    cx: &'a mut ExtCtxt<'b>,
    invocations: Vec<(Invocation, Option<Arc<SyntaxExtension>>)>,
    monotonic: bool,
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn cfg(&self) -> StripUnconfigured<'_> {
        StripUnconfigured {
            sess: self.cx.sess,
            features: Some(self.cx.ecfg.features),
            config_tokens: false,
            lint_node_id: self.cx.current_expansion.lint_node_id,
        }
    }

    fn collect(&mut self, fragment_kind: AstFragmentKind, kind: InvocationKind) -> AstFragment {
        let expn_id = LocalExpnId::fresh_empty();
        if matches!(kind, InvocationKind::GlobDelegation { .. }) {
            // In resolver we need to know which invocation ids are delegations early,
            // before their `ExpnData` is filled.
            self.cx.resolver.register_glob_delegation(expn_id);
        }
        let vis = kind.placeholder_visibility();
        self.invocations.push((
            Invocation {
                kind,
                fragment_kind,
                expansion_data: ExpansionData {
                    id: expn_id,
                    depth: self.cx.current_expansion.depth + 1,
                    ..self.cx.current_expansion.clone()
                },
            },
            None,
        ));
        placeholder(fragment_kind, NodeId::placeholder_from_expn_id(expn_id), vis)
    }

    fn collect_bang(&mut self, mac: Box<ast::MacCall>, kind: AstFragmentKind) -> AstFragment {
        // cache the macro call span so that it can be
        // easily adjusted for incremental compilation
        let span = mac.span();
        self.collect(kind, InvocationKind::Bang { mac, span })
    }

    fn collect_attr(
        &mut self,
        (attr, pos, derives): (ast::Attribute, usize, Vec<ast::Path>),
        item: Annotatable,
        kind: AstFragmentKind,
    ) -> AstFragment {
        self.collect(kind, InvocationKind::Attr { attr, pos, item, derives })
    }

    fn collect_glob_delegation(
        &mut self,
        item: Box<ast::AssocItem>,
        of_trait: bool,
        kind: AstFragmentKind,
    ) -> AstFragment {
        self.collect(kind, InvocationKind::GlobDelegation { item, of_trait })
    }

    /// If `item` is an attribute invocation, remove the attribute and return it together with
    /// its position and derives following it. We have to collect the derives in order to resolve
    /// legacy derive helpers (helpers written before derives that introduce them).
    fn take_first_attr(
        &self,
        item: &mut impl HasAttrs,
    ) -> Option<(ast::Attribute, usize, Vec<ast::Path>)> {
        let mut attr = None;

        let mut cfg_pos = None;
        let mut attr_pos = None;
        for (pos, attr) in item.attrs().iter().enumerate() {
            if !attr.is_doc_comment() && !self.cx.expanded_inert_attrs.is_marked(attr) {
                let name = attr.ident().map(|ident| ident.name);
                if name == Some(sym::cfg) || name == Some(sym::cfg_attr) {
                    cfg_pos = Some(pos); // a cfg attr found, no need to search anymore
                    break;
                } else if attr_pos.is_none()
                    && !name.is_some_and(rustc_feature::is_builtin_attr_name)
                {
                    attr_pos = Some(pos); // a non-cfg attr found, still may find a cfg attr
                }
            }
        }

        item.visit_attrs(|attrs| {
            attr = Some(match (cfg_pos, attr_pos) {
                (Some(pos), _) => (attrs.remove(pos), pos, Vec::new()),
                (_, Some(pos)) => {
                    let attr = attrs.remove(pos);
                    let following_derives = attrs[pos..]
                        .iter()
                        .filter(|a| a.has_name(sym::derive))
                        .flat_map(|a| a.meta_item_list().unwrap_or_default())
                        .filter_map(|meta_item_inner| match meta_item_inner {
                            MetaItemInner::MetaItem(ast::MetaItem {
                                kind: MetaItemKind::Word,
                                path,
                                ..
                            }) => Some(path),
                            _ => None,
                        })
                        .collect();

                    (attr, pos, following_derives)
                }
                _ => return,
            });
        });

        attr
    }

    // Detect use of feature-gated or invalid attributes on macro invocations
    // since they will not be detected after macro expansion.
    fn check_attributes(&self, attrs: &[ast::Attribute], call: &ast::MacCall) {
        let features = self.cx.ecfg.features;
        let mut attrs = attrs.iter().peekable();
        let mut span: Option<Span> = None;
        while let Some(attr) = attrs.next() {
            rustc_ast_passes::feature_gate::check_attribute(attr, self.cx.sess, features);
            validate_attr::check_attr(
                &self.cx.sess.psess,
                attr,
                self.cx.current_expansion.lint_node_id,
            );
            AttributeParser::parse_limited_all(
                self.cx.sess,
                slice::from_ref(attr),
                None,
                Target::MacroCall,
                call.span(),
                self.cx.current_expansion.lint_node_id,
                Some(self.cx.ecfg.features),
                ShouldEmit::ErrorsAndLints,
            );

            let current_span = if let Some(sp) = span { sp.to(attr.span) } else { attr.span };
            span = Some(current_span);

            if attrs.peek().is_some_and(|next_attr| next_attr.doc_str().is_some()) {
                continue;
            }

            if attr.is_doc_comment() {
                self.cx.sess.psess.buffer_lint(
                    UNUSED_DOC_COMMENTS,
                    current_span,
                    self.cx.current_expansion.lint_node_id,
                    crate::errors::MacroCallUnusedDocComment { span: attr.span },
                );
            } else if rustc_attr_parsing::is_builtin_attr(attr)
                && !AttributeParser::<Early>::is_parsed_attribute(&attr.path())
            {
                let attr_name = attr.ident().unwrap().name;
                // `#[cfg]` and `#[cfg_attr]` are special - they are
                // eagerly evaluated.
                if attr_name != sym::cfg_trace && attr_name != sym::cfg_attr_trace {
                    self.cx.sess.psess.buffer_lint(
                        UNUSED_ATTRIBUTES,
                        attr.span,
                        self.cx.current_expansion.lint_node_id,
                        crate::errors::UnusedBuiltinAttribute {
                            attr_name,
                            macro_name: pprust::path_to_string(&call.path),
                            invoc_span: call.path.span,
                            attr_span: attr.span,
                        },
                    );
                }
            }
        }
    }

    fn expand_cfg_true(
        &mut self,
        node: &mut (impl HasAttrs + HasNodeId),
        attr: ast::Attribute,
        pos: usize,
    ) -> EvalConfigResult {
        let res = self.cfg().cfg_true(&attr, node.node_id(), ShouldEmit::ErrorsAndLints);
        if res.as_bool() {
            // A trace attribute left in AST in place of the original `cfg` attribute.
            // It can later be used by lints or other diagnostics.
            let trace_attr = attr_into_trace(attr, sym::cfg_trace);
            node.visit_attrs(|attrs| attrs.insert(pos, trace_attr));
        }

        res
    }

    fn expand_cfg_attr(&self, node: &mut impl HasAttrs, attr: &ast::Attribute, pos: usize) {
        node.visit_attrs(|attrs| {
            // Repeated `insert` calls is inefficient, but the number of
            // insertions is almost always 0 or 1 in practice.
            for cfg in self.cfg().expand_cfg_attr(attr, false).into_iter().rev() {
                attrs.insert(pos, cfg)
            }
        });
    }

    fn flat_map_node<Node: InvocationCollectorNode<OutputTy: Default>>(
        &mut self,
        mut node: Node,
    ) -> Node::OutputTy {
        loop {
            return match self.take_first_attr(&mut node) {
                Some((attr, pos, derives)) => match attr.name() {
                    Some(sym::cfg) => {
                        let res = self.expand_cfg_true(&mut node, attr, pos);
                        match res {
                            EvalConfigResult::True => continue,
                            EvalConfigResult::False { reason, reason_span } => {
                                for ident in node.declared_idents() {
                                    self.cx.resolver.append_stripped_cfg_item(
                                        self.cx.current_expansion.lint_node_id,
                                        ident,
                                        reason.clone(),
                                        reason_span,
                                    )
                                }
                            }
                        }

                        Default::default()
                    }
                    Some(sym::cfg_attr) => {
                        self.expand_cfg_attr(&mut node, &attr, pos);
                        continue;
                    }
                    _ => {
                        Node::pre_flat_map_node_collect_attr(&self.cfg(), &attr);
                        self.collect_attr((attr, pos, derives), node.to_annotatable(), Node::KIND)
                            .make_ast::<Node>()
                    }
                },
                None if node.is_mac_call() => {
                    let (mac, attrs, add_semicolon) = node.take_mac_call();
                    self.check_attributes(&attrs, &mac);
                    let mut res = self.collect_bang(mac, Node::KIND).make_ast::<Node>();
                    Node::post_flat_map_node_collect_bang(&mut res, add_semicolon);
                    res
                }
                None if let Some((deleg, item)) = node.delegation() => {
                    let Some(suffixes) = &deleg.suffixes else {
                        let traitless_qself =
                            matches!(&deleg.qself, Some(qself) if qself.position == 0);
                        let (item, of_trait) = match node.to_annotatable() {
                            Annotatable::AssocItem(item, AssocCtxt::Impl { of_trait }) => {
                                (item, of_trait)
                            }
                            ann @ (Annotatable::Item(_)
                            | Annotatable::AssocItem(..)
                            | Annotatable::Stmt(_)) => {
                                let span = ann.span();
                                self.cx.dcx().emit_err(GlobDelegationOutsideImpls { span });
                                return Default::default();
                            }
                            _ => unreachable!(),
                        };
                        if traitless_qself {
                            let span = item.span;
                            self.cx.dcx().emit_err(GlobDelegationTraitlessQpath { span });
                            return Default::default();
                        }
                        return self
                            .collect_glob_delegation(item, of_trait, Node::KIND)
                            .make_ast::<Node>();
                    };

                    let single_delegations = build_single_delegations::<Node>(
                        self.cx, deleg, item, suffixes, item.span, false,
                    );
                    Node::flatten_outputs(single_delegations.map(|item| {
                        let mut item = Node::from_item(item);
                        assign_id!(self, item.node_id_mut(), || item.walk_flat_map(self))
                    }))
                }
                None => {
                    match Node::wrap_flat_map_node_walk_flat_map(node, self, |mut node, this| {
                        assign_id!(this, node.node_id_mut(), || node.walk_flat_map(this))
                    }) {
                        Ok(output) => output,
                        Err(returned_node) => {
                            node = returned_node;
                            continue;
                        }
                    }
                }
            };
        }
    }

    fn visit_node<Node: InvocationCollectorNode<OutputTy: Into<Node>> + DummyAstNode>(
        &mut self,
        node: &mut Node,
    ) {
        loop {
            return match self.take_first_attr(node) {
                Some((attr, pos, derives)) => match attr.name() {
                    Some(sym::cfg) => {
                        let span = attr.span;
                        if self.expand_cfg_true(node, attr, pos).as_bool() {
                            continue;
                        }

                        node.expand_cfg_false(self, pos, span);
                        continue;
                    }
                    Some(sym::cfg_attr) => {
                        self.expand_cfg_attr(node, &attr, pos);
                        continue;
                    }
                    _ => {
                        let n = mem::replace(node, Node::dummy());
                        *node = self
                            .collect_attr((attr, pos, derives), n.to_annotatable(), Node::KIND)
                            .make_ast::<Node>()
                            .into()
                    }
                },
                None if node.is_mac_call() => {
                    let n = mem::replace(node, Node::dummy());
                    let (mac, attrs, _) = n.take_mac_call();
                    self.check_attributes(&attrs, &mac);

                    *node = self.collect_bang(mac, Node::KIND).make_ast::<Node>().into()
                }
                None if node.delegation().is_some() => unreachable!(),
                None => {
                    assign_id!(self, node.node_id_mut(), || node.walk(self))
                }
            };
        }
    }
}

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn flat_map_item(&mut self, node: Box<ast::Item>) -> SmallVec<[Box<ast::Item>; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_assoc_item(
        &mut self,
        node: Box<ast::AssocItem>,
        ctxt: AssocCtxt,
    ) -> SmallVec<[Box<ast::AssocItem>; 1]> {
        match ctxt {
            AssocCtxt::Trait => self.flat_map_node(AstNodeWrapper::new(node, TraitItemTag)),
            AssocCtxt::Impl { of_trait: false } => {
                self.flat_map_node(AstNodeWrapper::new(node, ImplItemTag))
            }
            AssocCtxt::Impl { of_trait: true } => {
                self.flat_map_node(AstNodeWrapper::new(node, TraitImplItemTag))
            }
        }
    }

    fn flat_map_foreign_item(
        &mut self,
        node: Box<ast::ForeignItem>,
    ) -> SmallVec<[Box<ast::ForeignItem>; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_variant(&mut self, node: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_where_predicate(
        &mut self,
        node: ast::WherePredicate,
    ) -> SmallVec<[ast::WherePredicate; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_field_def(&mut self, node: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_pat_field(&mut self, node: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_expr_field(&mut self, node: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_param(&mut self, node: ast::Param) -> SmallVec<[ast::Param; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_generic_param(
        &mut self,
        node: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_arm(&mut self, node: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_stmt(&mut self, node: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        // FIXME: invocations in semicolon-less expressions positions are expanded as expressions,
        // changing that requires some compatibility measures.
        if node.is_expr() {
            // The only way that we can end up with a `MacCall` expression statement,
            // (as opposed to a `StmtKind::MacCall`) is if we have a macro as the
            // trailing expression in a block (e.g. `fn foo() { my_macro!() }`).
            // Record this information, so that we can report a more specific
            // `SEMICOLON_IN_EXPRESSIONS_FROM_MACROS` lint if needed.
            // See #78991 for an investigation of treating macros in this position
            // as statements, rather than expressions, during parsing.
            return match &node.kind {
                StmtKind::Expr(expr)
                    if matches!(**expr, ast::Expr { kind: ExprKind::MacCall(..), .. }) =>
                {
                    self.cx.current_expansion.is_trailing_mac = true;
                    // Don't use `assign_id` for this statement - it may get removed
                    // entirely due to a `#[cfg]` on the contained expression
                    let res = walk_flat_map_stmt(self, node);
                    self.cx.current_expansion.is_trailing_mac = false;
                    res
                }
                _ => walk_flat_map_stmt(self, node),
            };
        }

        self.flat_map_node(node)
    }

    fn visit_crate(&mut self, node: &mut ast::Crate) {
        self.visit_node(node)
    }

    fn visit_ty(&mut self, node: &mut ast::Ty) {
        self.visit_node(node)
    }

    fn visit_pat(&mut self, node: &mut ast::Pat) {
        self.visit_node(node)
    }

    fn visit_expr(&mut self, node: &mut ast::Expr) {
        // FIXME: Feature gating is performed inconsistently between `Expr` and `OptExpr`.
        if let Some(attr) = node.attrs.first() {
            self.cfg().maybe_emit_expr_attr_err(attr);
        }
        ensure_sufficient_stack(|| self.visit_node(node))
    }

    fn visit_method_receiver_expr(&mut self, node: &mut ast::Expr) {
        self.visit_node(AstNodeWrapper::from_mut(node, MethodReceiverTag))
    }

    fn filter_map_expr(&mut self, node: Box<ast::Expr>) -> Option<Box<ast::Expr>> {
        self.flat_map_node(AstNodeWrapper::new(node, OptExprTag))
    }

    fn visit_block(&mut self, node: &mut ast::Block) {
        let orig_dir_ownership = mem::replace(
            &mut self.cx.current_expansion.dir_ownership,
            DirOwnership::UnownedViaBlock,
        );
        walk_block(self, node);
        self.cx.current_expansion.dir_ownership = orig_dir_ownership;
    }

    fn visit_id(&mut self, id: &mut NodeId) {
        // We may have already assigned a `NodeId`
        // by calling `assign_id`
        if self.monotonic && *id == ast::DUMMY_NODE_ID {
            *id = self.cx.resolver.next_node_id();
        }
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: Symbol,
    pub features: &'feat Features,
    pub recursion_limit: Limit,
    pub trace_mac: bool,
    /// If false, strip `#[test]` nodes
    pub should_test: bool,
    /// If true, use verbose debugging for `proc_macro::Span`
    pub span_debug: bool,
    /// If true, show backtraces for proc-macro panics
    pub proc_macro_backtrace: bool,
}

impl ExpansionConfig<'_> {
    pub fn default(crate_name: Symbol, features: &Features) -> ExpansionConfig<'_> {
        ExpansionConfig {
            crate_name,
            features,
            // FIXME should this limit be configurable?
            recursion_limit: Limit::new(1024),
            trace_mac: false,
            should_test: false,
            span_debug: false,
            proc_macro_backtrace: false,
        }
    }
}
