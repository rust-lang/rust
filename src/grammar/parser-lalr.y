// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

%{
#define YYERROR_VERBOSE
#define YYSTYPE struct node *
struct node;
extern int yylex();
extern void yyerror(char const *s);
extern struct node *mk_node(char const *name, int n, ...);
extern struct node *mk_atom(char *text);
extern struct node *mk_none();
extern struct node *ext_node(struct node *nd, int n, ...);
extern void push_back(char c);
extern char *yytext;
%}
%debug

%token SHL
%token SHR
%token LE
%token EQEQ
%token NE
%token GE
%token ANDAND
%token OROR
%token SHLEQ
%token SHREQ
%token MINUSEQ
%token ANDEQ
%token OREQ
%token PLUSEQ
%token STAREQ
%token SLASHEQ
%token CARETEQ
%token PERCENTEQ
%token DOTDOT
%token DOTDOTDOT
%token MOD_SEP
%token RARROW
%token LARROW
%token FAT_ARROW
%token LIT_BYTE
%token LIT_CHAR
%token LIT_INTEGER
%token LIT_FLOAT
%token LIT_STR
%token LIT_STR_RAW
%token LIT_BYTE_STR
%token LIT_BYTE_STR_RAW
%token IDENT
%token UNDERSCORE
%token LIFETIME

// keywords
%token SELF
%token STATIC
%token AS
%token BREAK
%token CRATE
%token ELSE
%token ENUM
%token EXTERN
%token FALSE
%token FN
%token FOR
%token IF
%token IMPL
%token IN
%token LET
%token LOOP
%token MATCH
%token MOD
%token MOVE
%token MUT
%token PRIV
%token PUB
%token REF
%token RETURN
%token STRUCT
%token TRUE
%token TRAIT
%token TYPE
%token UNSAFE
%token USE
%token WHILE
%token CONTINUE
%token PROC
%token BOX
%token CONST
%token WHERE
%token TYPEOF
%token INNER_DOC_COMMENT
%token OUTER_DOC_COMMENT

%token SHEBANG
%token SHEBANG_LINE
%token STATIC_LIFETIME

 /*
   Quoting from the Bison manual:

   "Finally, the resolution of conflicts works by comparing the precedence
   of the rule being considered with that of the lookahead token. If the
   token's precedence is higher, the choice is to shift. If the rule's
   precedence is higher, the choice is to reduce. If they have equal
   precedence, the choice is made based on the associativity of that
   precedence level. The verbose output file made by ‘-v’ (see Invoking
   Bison) says how each conflict was resolved"
 */

// We expect no shift/reduce or reduce/reduce conflicts in this grammar;
// all potential ambiguities are scrutinized and eliminated manually.
%expect 0

// fake-precedence symbol to cause '|' bars in lambda context to parse
// at low precedence, permit things like |x| foo = bar, where '=' is
// otherwise lower-precedence than '|'. Also used for proc() to cause
// things like proc() a + b to parse as proc() { a + b }.
%precedence LAMBDA

%precedence SELF

// MUT should be lower precedence than IDENT so that in the pat rule,
// "& MUT pat" has higher precedence than "binding_mode ident [@ pat]"
%precedence MUT

// IDENT needs to be lower than '{' so that 'foo {' is shifted when
// trying to decide if we've got a struct-construction expr (esp. in
// contexts like 'if foo { .')
//
// IDENT also needs to be lower precedence than '<' so that '<' in
// 'foo:bar . <' is shifted (in a trait reference occurring in a
// bounds list), parsing as foo:(bar<baz>) rather than (foo:bar)<baz>.
%precedence IDENT

// A couple fake-precedence symbols to use in rules associated with +
// and < in trailing type contexts. These come up when you have a type
// in the RHS of operator-AS, such as "foo as bar<baz>". The "<" there
// has to be shifted so the parser keeps trying to parse a type, even
// though it might well consider reducing the type "bar" and then
// going on to "<" as a subsequent binop. The "+" case is with
// trailing type-bounds ("foo as bar:A+B"), for the same reason.
%precedence SHIFTPLUS

%precedence MOD_SEP
%precedence RARROW ':'

// In where clauses, "for" should have greater precedence when used as
// a higher ranked constraint than when used as the beginning of a
// for_in_type (which is a ty)
%precedence FORTYPE
%precedence FOR

// Binops & unops, and their precedences
%precedence BOX
%precedence BOXPLACE
%nonassoc DOTDOT

// RETURN needs to be lower-precedence than tokens that start
// prefix_exprs
%precedence RETURN

%right '=' SHLEQ SHREQ MINUSEQ ANDEQ OREQ PLUSEQ STAREQ SLASHEQ CARETEQ PERCENTEQ
%right LARROW
%left OROR
%left ANDAND
%left EQEQ NE
%left '<' '>' LE GE
%left '|'
%left '^'
%left '&'
%left SHL SHR
%left '+' '-'
%precedence AS
%left '*' '/' '%'
%precedence '!'

%precedence '{' '[' '(' '.'

%precedence RANGE

%start crate

%%

////////////////////////////////////////////////////////////////////////
// Part 1: Items and attributes
////////////////////////////////////////////////////////////////////////

crate
: maybe_shebang inner_attrs maybe_mod_items  { mk_node("crate", 2, $2, $3); }
| maybe_shebang maybe_mod_items  { mk_node("crate", 1, $2); }
;

maybe_shebang
: SHEBANG_LINE
| %empty
;

maybe_inner_attrs
: inner_attrs
| %empty                   { $$ = mk_none(); }
;

inner_attrs
: inner_attr               { $$ = mk_node("InnerAttrs", 1, $1); }
| inner_attrs inner_attr   { $$ = ext_node($1, 1, $2); }
;

inner_attr
: SHEBANG '[' meta_item ']'   { $$ = mk_node("InnerAttr", 1, $3); }
| INNER_DOC_COMMENT           { $$ = mk_node("InnerAttr", 1, mk_node("doc-comment", 1, mk_atom(yytext))); }
;

maybe_outer_attrs
: outer_attrs
| %empty                   { $$ = mk_none(); }
;

outer_attrs
: outer_attr               { $$ = mk_node("OuterAttrs", 1, $1); }
| outer_attrs outer_attr   { $$ = ext_node($1, 1, $2); }
;

outer_attr
: '#' '[' meta_item ']'    { $$ = $3; }
| OUTER_DOC_COMMENT        { $$ = mk_node("doc-comment", 1, mk_atom(yytext)); }
;

meta_item
: ident                      { $$ = mk_node("MetaWord", 1, $1); }
| ident '=' lit              { $$ = mk_node("MetaNameValue", 2, $1, $3); }
| ident '(' meta_seq ')'     { $$ = mk_node("MetaList", 2, $1, $3); }
| ident '(' meta_seq ',' ')' { $$ = mk_node("MetaList", 2, $1, $3); }
;

meta_seq
: %empty                   { $$ = mk_none(); }
| meta_item                { $$ = mk_node("MetaItems", 1, $1); }
| meta_seq ',' meta_item   { $$ = ext_node($1, 1, $3); }
;

maybe_mod_items
: mod_items
| %empty             { $$ = mk_none(); }
;

mod_items
: mod_item                               { $$ = mk_node("Items", 1, $1); }
| mod_items mod_item                     { $$ = ext_node($1, 1, $2); }
;

attrs_and_vis
: maybe_outer_attrs visibility           { $$ = mk_node("AttrsAndVis", 2, $1, $2); }
;

mod_item
: attrs_and_vis item    { $$ = mk_node("Item", 2, $1, $2); }
;

// items that can appear outside of a fn block
item
: stmt_item
| item_macro
;

// items that can appear in "stmts"
stmt_item
: item_static
| item_const
| item_type
| block_item
| view_item
;

item_static
: STATIC ident ':' ty '=' expr ';'  { $$ = mk_node("ItemStatic", 3, $2, $4, $6); }
| STATIC MUT ident ':' ty '=' expr ';'  { $$ = mk_node("ItemStatic", 3, $3, $5, $7); }
;

item_const
: CONST ident ':' ty '=' expr ';'  { $$ = mk_node("ItemConst", 3, $2, $4, $6); }
;

item_macro
: path_expr '!' maybe_ident parens_delimited_token_trees ';'  { $$ = mk_node("ItemMacro", 3, $1, $3, $4); }
| path_expr '!' maybe_ident braces_delimited_token_trees      { $$ = mk_node("ItemMacro", 3, $1, $3, $4); }
| path_expr '!' maybe_ident brackets_delimited_token_trees ';'{ $$ = mk_node("ItemMacro", 3, $1, $3, $4); }
;

view_item
: use_item
| extern_fn_item
| EXTERN CRATE ident ';'                      { $$ = mk_node("ViewItemExternCrate", 1, $3); }
| EXTERN CRATE ident AS ident ';'             { $$ = mk_node("ViewItemExternCrate", 2, $3, $5); }
;

extern_fn_item
: EXTERN maybe_abi item_fn                    { $$ = mk_node("ViewItemExternFn", 2, $2, $3); }
;

use_item
: USE view_path ';'                           { $$ = mk_node("ViewItemUse", 1, $2); }
;

view_path
: path_no_types_allowed                                    { $$ = mk_node("ViewPathSimple", 1, $1); }
| path_no_types_allowed MOD_SEP '{'                '}'     { $$ = mk_node("ViewPathList", 2, $1, mk_atom("ViewPathListEmpty")); }
|                       MOD_SEP '{'                '}'     { $$ = mk_node("ViewPathList", 1, mk_atom("ViewPathListEmpty")); }
| path_no_types_allowed MOD_SEP '{' idents_or_self '}'     { $$ = mk_node("ViewPathList", 2, $1, $4); }
|                       MOD_SEP '{' idents_or_self '}'     { $$ = mk_node("ViewPathList", 1, $3); }
| path_no_types_allowed MOD_SEP '{' idents_or_self ',' '}' { $$ = mk_node("ViewPathList", 2, $1, $4); }
|                       MOD_SEP '{' idents_or_self ',' '}' { $$ = mk_node("ViewPathList", 1, $3); }
| path_no_types_allowed MOD_SEP '*'                        { $$ = mk_node("ViewPathGlob", 1, $1); }
|                               '{'                '}'     { $$ = mk_atom("ViewPathListEmpty"); }
|                               '{' idents_or_self '}'     { $$ = mk_node("ViewPathList", 1, $2); }
|                               '{' idents_or_self ',' '}' { $$ = mk_node("ViewPathList", 1, $2); }
| path_no_types_allowed AS ident                           { $$ = mk_node("ViewPathSimple", 2, $1, $3); }
;

block_item
: item_fn
| item_unsafe_fn
| item_mod
| item_foreign_mod          { $$ = mk_node("ItemForeignMod", 1, $1); }
| item_struct
| item_enum
| item_trait
| item_impl
;

maybe_ty_ascription
: ':' ty_sum { $$ = $2; }
| %empty { $$ = mk_none(); }
;

maybe_init_expr
: '=' expr { $$ = $2; }
| %empty   { $$ = mk_none(); }
;

// structs
item_struct
: STRUCT ident generic_params maybe_where_clause struct_decl_args
{
  $$ = mk_node("ItemStruct", 4, $2, $3, $4, $5);
}
| STRUCT ident generic_params struct_tuple_args maybe_where_clause ';'
{
  $$ = mk_node("ItemStruct", 4, $2, $3, $4, $5);
}
| STRUCT ident generic_params maybe_where_clause ';'
{
  $$ = mk_node("ItemStruct", 3, $2, $3, $4);
}
;

struct_decl_args
: '{' struct_decl_fields '}'                  { $$ = $2; }
| '{' struct_decl_fields ',' '}'              { $$ = $2; }
;

struct_tuple_args
: '(' struct_tuple_fields ')'                 { $$ = $2; }
| '(' struct_tuple_fields ',' ')'             { $$ = $2; }
;

struct_decl_fields
: struct_decl_field                           { $$ = mk_node("StructFields", 1, $1); }
| struct_decl_fields ',' struct_decl_field    { $$ = ext_node($1, 1, $3); }
| %empty                                      { $$ = mk_none(); }
;

struct_decl_field
: attrs_and_vis ident ':' ty_sum              { $$ = mk_node("StructField", 3, $1, $2, $4); }
;

struct_tuple_fields
: struct_tuple_field                          { $$ = mk_node("StructFields", 1, $1); }
| struct_tuple_fields ',' struct_tuple_field  { $$ = ext_node($1, 1, $3); }
;

struct_tuple_field
: attrs_and_vis ty_sum                    { $$ = mk_node("StructField", 2, $1, $2); }
;

// enums
item_enum
: ENUM ident generic_params maybe_where_clause '{' enum_defs '}'     { $$ = mk_node("ItemEnum", 0); }
| ENUM ident generic_params maybe_where_clause '{' enum_defs ',' '}' { $$ = mk_node("ItemEnum", 0); }
;

enum_defs
: enum_def               { $$ = mk_node("EnumDefs", 1, $1); }
| enum_defs ',' enum_def { $$ = ext_node($1, 1, $3); }
| %empty                 { $$ = mk_none(); }
;

enum_def
: attrs_and_vis ident enum_args { $$ = mk_node("EnumDef", 3, $1, $2, $3); }
;

enum_args
: '{' struct_decl_fields '}'     { $$ = mk_node("EnumArgs", 1, $2); }
| '{' struct_decl_fields ',' '}' { $$ = mk_node("EnumArgs", 1, $2); }
| '(' maybe_ty_sums ')'          { $$ = mk_node("EnumArgs", 1, $2); }
| '=' expr                       { $$ = mk_node("EnumArgs", 1, $2); }
| %empty                         { $$ = mk_none(); }
;

item_mod
: MOD ident ';'                                 { $$ = mk_node("ItemMod", 1, $2); }
| MOD ident '{' maybe_mod_items '}'             { $$ = mk_node("ItemMod", 2, $2, $4); }
| MOD ident '{' inner_attrs maybe_mod_items '}' { $$ = mk_node("ItemMod", 3, $2, $4, $5); }
;

item_foreign_mod
: EXTERN maybe_abi '{' maybe_foreign_items '}'             { $$ = mk_node("ItemForeignMod", 1, $4); }
| EXTERN maybe_abi '{' inner_attrs maybe_foreign_items '}' { $$ = mk_node("ItemForeignMod", 2, $4, $5); }
;

maybe_abi
: str
| %empty { $$ = mk_none(); }
;

maybe_foreign_items
: foreign_items
| %empty { $$ = mk_none(); }
;

foreign_items
: foreign_item               { $$ = mk_node("ForeignItems", 1, $1); }
| foreign_items foreign_item { $$ = ext_node($1, 1, $2); }
;

foreign_item
: attrs_and_vis STATIC item_foreign_static { $$ = mk_node("ForeignItem", 2, $1, $3); }
| attrs_and_vis item_foreign_fn            { $$ = mk_node("ForeignItem", 2, $1, $2); }
| attrs_and_vis UNSAFE item_foreign_fn     { $$ = mk_node("ForeignItem", 2, $1, $3); }
;

item_foreign_static
: maybe_mut ident ':' ty ';'               { $$ = mk_node("StaticItem", 3, $1, $2, $4); }
;

item_foreign_fn
: FN ident generic_params fn_decl_allow_variadic maybe_where_clause ';' { $$ = mk_node("ForeignFn", 4, $2, $3, $4, $5); }
;

fn_decl_allow_variadic
: fn_params_allow_variadic ret_ty { $$ = mk_node("FnDecl", 2, $1, $2); }
;

fn_params_allow_variadic
: '(' ')'                      { $$ = mk_none(); }
| '(' params ')'               { $$ = $2; }
| '(' params ',' ')'           { $$ = $2; }
| '(' params ',' DOTDOTDOT ')' { $$ = $2; }
;

visibility
: PUB      { $$ = mk_atom("Public"); }
| %empty   { $$ = mk_atom("Inherited"); }
;

idents_or_self
: ident_or_self                    { $$ = mk_node("IdentsOrSelf", 1, $1); }
| ident_or_self AS ident           { $$ = mk_node("IdentsOrSelf", 2, $1, $3); }
| idents_or_self ',' ident_or_self { $$ = ext_node($1, 1, $3); }
;

ident_or_self
: ident
| SELF  { $$ = mk_atom(yytext); }
;

item_type
: TYPE ident generic_params maybe_where_clause '=' ty_sum ';'  { $$ = mk_node("ItemTy", 4, $2, $3, $4, $6); }
;

for_sized
: FOR '?' ident { $$ = mk_node("ForSized", 1, $3); }
| FOR ident '?' { $$ = mk_node("ForSized", 1, $2); }
| %empty        { $$ = mk_none(); }
;

item_trait
: maybe_unsafe TRAIT ident generic_params for_sized maybe_ty_param_bounds maybe_where_clause '{' maybe_trait_items '}'
{
  $$ = mk_node("ItemTrait", 7, $1, $3, $4, $5, $6, $7, $9);
}
;

maybe_trait_items
: trait_items
| %empty { $$ = mk_none(); }
;

trait_items
: trait_item               { $$ = mk_node("TraitItems", 1, $1); }
| trait_items trait_item   { $$ = ext_node($1, 1, $2); }
;

trait_item
: trait_const
| trait_type
| trait_method
;

trait_const
: maybe_outer_attrs CONST ident maybe_ty_ascription maybe_const_default ';' { $$ = mk_node("ConstTraitItem", 4, $1, $3, $4, $5); }
;

maybe_const_default
: '=' expr { $$ = mk_node("ConstDefault", 1, $2); }
| %empty   { $$ = mk_none(); }
;

trait_type
: maybe_outer_attrs TYPE ty_param ';' { $$ = mk_node("TypeTraitItem", 2, $1, $3); }
;

maybe_unsafe
: UNSAFE { $$ = mk_atom("Unsafe"); }
| %empty { $$ = mk_none(); }
;

trait_method
: type_method { $$ = mk_node("Required", 1, $1); }
| method      { $$ = mk_node("Provided", 1, $1); }
;

type_method
: attrs_and_vis maybe_unsafe FN ident generic_params fn_decl_with_self_allow_anon_params maybe_where_clause ';'
{
  $$ = mk_node("TypeMethod", 6, $1, $2, $4, $5, $6, $7);
}
| attrs_and_vis maybe_unsafe EXTERN maybe_abi FN ident generic_params fn_decl_with_self_allow_anon_params maybe_where_clause ';'
{
  $$ = mk_node("TypeMethod", 7, $1, $2, $4, $6, $7, $8, $9);
}
;

method
: attrs_and_vis maybe_unsafe FN ident generic_params fn_decl_with_self_allow_anon_params maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("Method", 7, $1, $2, $4, $5, $6, $7, $8);
}
| attrs_and_vis maybe_unsafe EXTERN maybe_abi FN ident generic_params fn_decl_with_self_allow_anon_params maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("Method", 8, $1, $2, $4, $6, $7, $8, $9, $10);
}
;

impl_method
: attrs_and_vis maybe_unsafe FN ident generic_params fn_decl_with_self maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("Method", 7, $1, $2, $4, $5, $6, $7, $8);
}
| attrs_and_vis maybe_unsafe EXTERN maybe_abi FN ident generic_params fn_decl_with_self maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("Method", 8, $1, $2, $4, $6, $7, $8, $9, $10);
}
;

// There are two forms of impl:
//
// impl (<...>)? TY { ... }
// impl (<...>)? TRAIT for TY { ... }
//
// Unfortunately since TY can begin with '<' itself -- as part of a
// TyQualifiedPath type -- there's an s/r conflict when we see '<' after IMPL:
// should we reduce one of the early rules of TY (such as maybe_once)
// or shall we continue shifting into the generic_params list for the
// impl?
//
// The production parser disambiguates a different case here by
// permitting / requiring the user to provide parens around types when
// they are ambiguous with traits. We do the same here, regrettably,
// by splitting ty into ty and ty_prim.
item_impl
: maybe_unsafe IMPL generic_params ty_prim_sum maybe_where_clause '{' maybe_inner_attrs maybe_impl_items '}'
{
  $$ = mk_node("ItemImpl", 6, $1, $3, $4, $5, $7, $8);
}
| maybe_unsafe IMPL generic_params '(' ty ')' maybe_where_clause '{' maybe_inner_attrs maybe_impl_items '}'
{
  $$ = mk_node("ItemImpl", 6, $1, $3, 5, $6, $9, $10);
}
| maybe_unsafe IMPL generic_params trait_ref FOR ty_sum maybe_where_clause '{' maybe_inner_attrs maybe_impl_items '}'
{
  $$ = mk_node("ItemImpl", 6, $3, $4, $6, $7, $9, $10);
}
| maybe_unsafe IMPL generic_params '!' trait_ref FOR ty_sum maybe_where_clause '{' maybe_inner_attrs maybe_impl_items '}'
{
  $$ = mk_node("ItemImplNeg", 7, $1, $3, $5, $7, $8, $10, $11);
}
| maybe_unsafe IMPL generic_params trait_ref FOR DOTDOT '{' '}'
{
  $$ = mk_node("ItemImplDefault", 3, $1, $3, $4);
}
| maybe_unsafe IMPL generic_params '!' trait_ref FOR DOTDOT '{' '}'
{
  $$ = mk_node("ItemImplDefaultNeg", 3, $1, $3, $4);
}
;

maybe_impl_items
: impl_items
| %empty { $$ = mk_none(); }
;

impl_items
: impl_item               { $$ = mk_node("ImplItems", 1, $1); }
| impl_item impl_items    { $$ = ext_node($1, 1, $2); }
;

impl_item
: impl_method
| attrs_and_vis item_macro { $$ = mk_node("ImplMacroItem", 2, $1, $2); }
| impl_const
| impl_type
;

impl_const
: attrs_and_vis item_const { $$ = mk_node("ImplConst", 1, $1, $2); }
;

impl_type
: attrs_and_vis TYPE ident generic_params '=' ty_sum ';'  { $$ = mk_node("ImplType", 4, $1, $3, $4, $6); }
;

item_fn
: FN ident generic_params fn_decl maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("ItemFn", 5, $2, $3, $4, $5, $6);
}
;

item_unsafe_fn
: UNSAFE FN ident generic_params fn_decl maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("ItemUnsafeFn", 5, $3, $4, $5, $6, $7);
}
| UNSAFE EXTERN maybe_abi FN ident generic_params fn_decl maybe_where_clause inner_attrs_and_block
{
  $$ = mk_node("ItemUnsafeFn", 6, $3, $5, $6, $7, $8, $9);
}
;

fn_decl
: fn_params ret_ty   { $$ = mk_node("FnDecl", 2, $1, $2); }
;

fn_decl_with_self
: fn_params_with_self ret_ty   { $$ = mk_node("FnDecl", 2, $1, $2); }
;

fn_decl_with_self_allow_anon_params
: fn_anon_params_with_self ret_ty   { $$ = mk_node("FnDecl", 2, $1, $2); }
;

fn_params
: '(' maybe_params ')'  { $$ = $2; }
;

fn_anon_params
: '(' anon_param anon_params_allow_variadic_tail ')' { $$ = ext_node($2, 1, $3); }
| '(' ')'                                            { $$ = mk_none(); }
;

fn_params_with_self
: '(' maybe_mut SELF maybe_ty_ascription maybe_comma_params ')'              { $$ = mk_node("SelfValue", 3, $2, $4, $5); }
| '(' '&' maybe_mut SELF maybe_ty_ascription maybe_comma_params ')'          { $$ = mk_node("SelfRegion", 3, $3, $5, $6); }
| '(' '&' lifetime maybe_mut SELF maybe_ty_ascription maybe_comma_params ')' { $$ = mk_node("SelfRegion", 4, $3, $4, $6, $7); }
| '(' maybe_params ')'                                                       { $$ = mk_node("SelfStatic", 1, $2); }
;

fn_anon_params_with_self
: '(' maybe_mut SELF maybe_ty_ascription maybe_comma_anon_params ')'              { $$ = mk_node("SelfValue", 3, $2, $4, $5); }
| '(' '&' maybe_mut SELF maybe_ty_ascription maybe_comma_anon_params ')'          { $$ = mk_node("SelfRegion", 3, $3, $5, $6); }
| '(' '&' lifetime maybe_mut SELF maybe_ty_ascription maybe_comma_anon_params ')' { $$ = mk_node("SelfRegion", 4, $3, $4, $6, $7); }
| '(' maybe_anon_params ')'                                                       { $$ = mk_node("SelfStatic", 1, $2); }
;

maybe_params
: params
| params ','
| %empty  { $$ = mk_none(); }
;

params
: param                { $$ = mk_node("Args", 1, $1); }
| params ',' param     { $$ = ext_node($1, 1, $3); }
;

param
: pat ':' ty_sum   { $$ = mk_node("Arg", 2, $1, $3); }
;

inferrable_params
: inferrable_param                       { $$ = mk_node("InferrableParams", 1, $1); }
| inferrable_params ',' inferrable_param { $$ = ext_node($1, 1, $3); }
;

inferrable_param
: pat maybe_ty_ascription { $$ = mk_node("InferrableParam", 2, $1, $2); }
;

maybe_unboxed_closure_kind
: %empty
| ':'
| '&' maybe_mut ':'
;

maybe_comma_params
: ','            { $$ = mk_none(); }
| ',' params     { $$ = $2; }
| ',' params ',' { $$ = $2; }
| %empty         { $$ = mk_none(); }
;

maybe_comma_anon_params
: ','                 { $$ = mk_none(); }
| ',' anon_params     { $$ = $2; }
| ',' anon_params ',' { $$ = $2; }
| %empty              { $$ = mk_none(); }
;

maybe_anon_params
: anon_params
| anon_params ','
| %empty      { $$ = mk_none(); }
;

anon_params
: anon_param                 { $$ = mk_node("Args", 1, $1); }
| anon_params ',' anon_param { $$ = ext_node($1, 1, $3); }
;

// anon means it's allowed to be anonymous (type-only), but it can
// still have a name
anon_param
: named_arg ':' ty   { $$ = mk_node("Arg", 2, $1, $3); }
| ty
;

anon_params_allow_variadic_tail
: ',' DOTDOTDOT                                  { $$ = mk_none(); }
| ',' anon_param anon_params_allow_variadic_tail { $$ = mk_node("Args", 2, $2, $3); }
| %empty                                         { $$ = mk_none(); }
;

named_arg
: ident
| UNDERSCORE        { $$ = mk_atom("PatWild"); }
| '&' ident         { $$ = $2; }
| '&' UNDERSCORE    { $$ = mk_atom("PatWild"); }
| ANDAND ident      { $$ = $2; }
| ANDAND UNDERSCORE { $$ = mk_atom("PatWild"); }
| MUT ident         { $$ = $2; }
;

ret_ty
: RARROW '!'         { $$ = mk_none(); }
| RARROW ty          { $$ = mk_node("ret-ty", 1, $2); }
| %prec IDENT %empty { $$ = mk_none(); }
;

generic_params
: '<' lifetimes '>'                   { $$ = mk_node("Generics", 2, $2, mk_none()); }
| '<' lifetimes ',' '>'               { $$ = mk_node("Generics", 2, $2, mk_none()); }
| '<' lifetimes SHR                   { push_back('>'); $$ = mk_node("Generics", 2, $2, mk_none()); }
| '<' lifetimes ',' SHR               { push_back('>'); $$ = mk_node("Generics", 2, $2, mk_none()); }
| '<' lifetimes ',' ty_params '>'     { $$ = mk_node("Generics", 2, $2, $4); }
| '<' lifetimes ',' ty_params ',' '>' { $$ = mk_node("Generics", 2, $2, $4); }
| '<' lifetimes ',' ty_params SHR     { push_back('>'); $$ = mk_node("Generics", 2, $2, $4); }
| '<' lifetimes ',' ty_params ',' SHR { push_back('>'); $$ = mk_node("Generics", 2, $2, $4); }
| '<' ty_params '>'                   { $$ = mk_node("Generics", 2, mk_none(), $2); }
| '<' ty_params ',' '>'               { $$ = mk_node("Generics", 2, mk_none(), $2); }
| '<' ty_params SHR                   { push_back('>'); $$ = mk_node("Generics", 2, mk_none(), $2); }
| '<' ty_params ',' SHR               { push_back('>'); $$ = mk_node("Generics", 2, mk_none(), $2); }
| %empty                              { $$ = mk_none(); }
;

maybe_where_clause
: %empty                              { $$ = mk_none(); }
| where_clause
;

where_clause
: WHERE where_predicates              { $$ = mk_node("WhereClause", 1, $2); }
| WHERE where_predicates ','          { $$ = mk_node("WhereClause", 1, $2); }
;

where_predicates
: where_predicate                      { $$ = mk_node("WherePredicates", 1, $1); }
| where_predicates ',' where_predicate { $$ = ext_node($1, 1, $3); }
;

where_predicate
: maybe_for_lifetimes lifetime ':' bounds    { $$ = mk_node("WherePredicate", 3, $1, $2, $4); }
| maybe_for_lifetimes ty ':' ty_param_bounds { $$ = mk_node("WherePredicate", 3, $1, $2, $4); }
;

maybe_for_lifetimes
: FOR '<' lifetimes '>' { $$ = mk_none(); }
| %prec FORTYPE %empty  { $$ = mk_none(); }

ty_params
: ty_param               { $$ = mk_node("TyParams", 1, $1); }
| ty_params ',' ty_param { $$ = ext_node($1, 1, $3); }
;

// A path with no type parameters; e.g. `foo::bar::Baz`
//
// These show up in 'use' view-items, because these are processed
// without respect to types.
path_no_types_allowed
: ident                               { $$ = mk_node("ViewPath", 1, $1); }
| MOD_SEP ident                       { $$ = mk_node("ViewPath", 1, $2); }
| SELF                                { $$ = mk_node("ViewPath", 1, mk_atom("Self")); }
| MOD_SEP SELF                        { $$ = mk_node("ViewPath", 1, mk_atom("Self")); }
| path_no_types_allowed MOD_SEP ident { $$ = ext_node($1, 1, $3); }
;

// A path with a lifetime and type parameters, with no double colons
// before the type parameters; e.g. `foo::bar<'a>::Baz<T>`
//
// These show up in "trait references", the components of
// type-parameter bounds lists, as well as in the prefix of the
// path_generic_args_and_bounds rule, which is the full form of a
// named typed expression.
//
// They do not have (nor need) an extra '::' before '<' because
// unlike in expr context, there are no "less-than" type exprs to
// be ambiguous with.
path_generic_args_without_colons
: %prec IDENT
  ident                                                                       { $$ = mk_node("components", 1, $1); }
| %prec IDENT
  ident generic_args                                                          { $$ = mk_node("components", 2, $1, $2); }
| %prec IDENT
  ident '(' maybe_ty_sums ')' ret_ty                                          { $$ = mk_node("components", 2, $1, $3); }
| %prec IDENT
  path_generic_args_without_colons MOD_SEP ident                              { $$ = ext_node($1, 1, $3); }
| %prec IDENT
  path_generic_args_without_colons MOD_SEP ident generic_args                 { $$ = ext_node($1, 2, $3, $4); }
| %prec IDENT
  path_generic_args_without_colons MOD_SEP ident '(' maybe_ty_sums ')' ret_ty { $$ = ext_node($1, 2, $3, $5); }
;

generic_args
: '<' generic_values '>'   { $$ = $2; }
| '<' generic_values SHR   { push_back('>'); $$ = $2; }
| '<' generic_values GE    { push_back('='); $$ = $2; }
| '<' generic_values SHREQ { push_back('>'); push_back('='); $$ = $2; }
// If generic_args starts with "<<", the first arg must be a
// TyQualifiedPath because that's the only type that can start with a
// '<'. This rule parses that as the first ty_sum and then continues
// with the rest of generic_values.
| SHL ty_qualified_path_and_generic_values '>'   { $$ = $2; }
| SHL ty_qualified_path_and_generic_values SHR   { push_back('>'); $$ = $2; }
| SHL ty_qualified_path_and_generic_values GE    { push_back('='); $$ = $2; }
| SHL ty_qualified_path_and_generic_values SHREQ { push_back('>'); push_back('='); $$ = $2; }
;

generic_values
: maybe_lifetimes maybe_ty_sums_and_or_bindings { $$ = mk_node("GenericValues", 2, $1, $2); }
;

maybe_ty_sums_and_or_bindings
: ty_sums
| ty_sums ','
| ty_sums ',' bindings { $$ = mk_node("TySumsAndBindings", 2, $1, $3); }
| bindings
| bindings ','
| %empty               { $$ = mk_none(); }
;

maybe_bindings
: ',' bindings { $$ = $2; }
| %empty       { $$ = mk_none(); }
;

////////////////////////////////////////////////////////////////////////
// Part 2: Patterns
////////////////////////////////////////////////////////////////////////

pat
: UNDERSCORE                                      { $$ = mk_atom("PatWild"); }
| '&' pat                                         { $$ = mk_node("PatRegion", 1, $2); }
| '&' MUT pat                                     { $$ = mk_node("PatRegion", 1, $3); }
| ANDAND pat                                      { $$ = mk_node("PatRegion", 1, mk_node("PatRegion", 1, $2)); }
| '(' ')'                                         { $$ = mk_atom("PatUnit"); }
| '(' pat_tup ')'                                 { $$ = mk_node("PatTup", 1, $2); }
| '(' pat_tup ',' ')'                             { $$ = mk_node("PatTup", 1, $2); }
| '[' pat_vec ']'                                 { $$ = mk_node("PatVec", 1, $2); }
| lit_or_path
| lit_or_path DOTDOTDOT lit_or_path               { $$ = mk_node("PatRange", 2, $1, $3); }
| path_expr '{' pat_struct '}'                    { $$ = mk_node("PatStruct", 2, $1, $3); }
| path_expr '(' DOTDOT ')'                        { $$ = mk_node("PatEnum", 1, $1); }
| path_expr '(' pat_tup ')'                       { $$ = mk_node("PatEnum", 2, $1, $3); }
| path_expr '!' maybe_ident delimited_token_trees { $$ = mk_node("PatMac", 3, $1, $3, $4); }
| binding_mode ident                              { $$ = mk_node("PatIdent", 2, $1, $2); }
|              ident '@' pat                      { $$ = mk_node("PatIdent", 3, mk_node("BindByValue", 1, mk_atom("MutImmutable")), $1, $3); }
| binding_mode ident '@' pat                      { $$ = mk_node("PatIdent", 3, $1, $2, $4); }
| BOX pat                                         { $$ = mk_node("PatUniq", 1, $2); }
| '<' ty_sum maybe_as_trait_ref '>' MOD_SEP ident { $$ = mk_node("PatQualifiedPath", 3, $2, $3, $6); }
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident maybe_as_trait_ref '>' MOD_SEP ident
{
  $$ = mk_node("PatQualifiedPath", 3, mk_node("PatQualifiedPath", 3, $2, $3, $6), $7, $10);
}
;

pats_or
: pat              { $$ = mk_node("Pats", 1, $1); }
| pats_or '|' pat  { $$ = ext_node($1, 1, $3); }
;

binding_mode
: REF         { $$ = mk_node("BindByRef", 1, mk_atom("MutImmutable")); }
| REF MUT     { $$ = mk_node("BindByRef", 1, mk_atom("MutMutable")); }
| MUT         { $$ = mk_node("BindByValue", 1, mk_atom("MutMutable")); }
;

lit_or_path
: path_expr    { $$ = mk_node("PatLit", 1, $1); }
| lit          { $$ = mk_node("PatLit", 1, $1); }
| '-' lit      { $$ = mk_node("PatLit", 1, $2); }
;

pat_field
:                  ident        { $$ = mk_node("PatField", 1, $1); }
|     binding_mode ident        { $$ = mk_node("PatField", 2, $1, $2); }
| BOX              ident        { $$ = mk_node("PatField", 2, mk_atom("box"), $2); }
| BOX binding_mode ident        { $$ = mk_node("PatField", 3, mk_atom("box"), $2, $3); }
|              ident ':' pat    { $$ = mk_node("PatField", 2, $1, $3); }
| binding_mode ident ':' pat    { $$ = mk_node("PatField", 3, $1, $2, $4); }
;

pat_fields
: pat_field                  { $$ = mk_node("PatFields", 1, $1); }
| pat_fields ',' pat_field   { $$ = ext_node($1, 1, $3); }
;

pat_struct
: pat_fields                 { $$ = mk_node("PatStruct", 2, $1, mk_atom("false")); }
| pat_fields ','             { $$ = mk_node("PatStruct", 2, $1, mk_atom("false")); }
| pat_fields ',' DOTDOT      { $$ = mk_node("PatStruct", 2, $1, mk_atom("true")); }
| DOTDOT                     { $$ = mk_node("PatStruct", 1, mk_atom("true")); }
;

pat_tup
: pat               { $$ = mk_node("pat_tup", 1, $1); }
| pat_tup ',' pat   { $$ = ext_node($1, 1, $3); }
;

pat_vec
: pat_vec_elts                                  { $$ = mk_node("PatVec", 2, $1, mk_none()); }
| pat_vec_elts                             ','  { $$ = mk_node("PatVec", 2, $1, mk_none()); }
| pat_vec_elts     DOTDOT                       { $$ = mk_node("PatVec", 2, $1, mk_none()); }
| pat_vec_elts ',' DOTDOT                       { $$ = mk_node("PatVec", 2, $1, mk_none()); }
| pat_vec_elts     DOTDOT ',' pat_vec_elts      { $$ = mk_node("PatVec", 2, $1, $4); }
| pat_vec_elts     DOTDOT ',' pat_vec_elts ','  { $$ = mk_node("PatVec", 2, $1, $4); }
| pat_vec_elts ',' DOTDOT ',' pat_vec_elts      { $$ = mk_node("PatVec", 2, $1, $5); }
| pat_vec_elts ',' DOTDOT ',' pat_vec_elts ','  { $$ = mk_node("PatVec", 2, $1, $5); }
|                  DOTDOT ',' pat_vec_elts      { $$ = mk_node("PatVec", 2, mk_none(), $3); }
|                  DOTDOT ',' pat_vec_elts ','  { $$ = mk_node("PatVec", 2, mk_none(), $3); }
|                  DOTDOT                       { $$ = mk_node("PatVec", 2, mk_none(), mk_none()); }
| %empty                                        { $$ = mk_node("PatVec", 2, mk_none(), mk_none()); }
;

pat_vec_elts
: pat                    { $$ = mk_node("PatVecElts", 1, $1); }
| pat_vec_elts ',' pat   { $$ = ext_node($1, 1, $3); }
;

////////////////////////////////////////////////////////////////////////
// Part 3: Types
////////////////////////////////////////////////////////////////////////

ty
: ty_prim
| ty_closure
| '<' ty_sum maybe_as_trait_ref '>' MOD_SEP ident                                      { $$ = mk_node("TyQualifiedPath", 3, $2, $3, $6); }
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident maybe_as_trait_ref '>' MOD_SEP ident { $$ = mk_node("TyQualifiedPath", 3, mk_node("TyQualifiedPath", 3, $2, $3, $6), $7, $10); }
| '(' ty_sums ')'                                                                      { $$ = mk_node("TyTup", 1, $2); }
| '(' ty_sums ',' ')'                                                                  { $$ = mk_node("TyTup", 1, $2); }
| '(' ')'                                                                              { $$ = mk_atom("TyNil"); }
;

ty_prim
: %prec IDENT path_generic_args_without_colons              { $$ = mk_node("TyPath", 2, mk_node("global", 1, mk_atom("false")), $1); }
| %prec IDENT MOD_SEP path_generic_args_without_colons      { $$ = mk_node("TyPath", 2, mk_node("global", 1, mk_atom("true")), $2); }
| %prec IDENT SELF MOD_SEP path_generic_args_without_colons { $$ = mk_node("TyPath", 2, mk_node("self", 1, mk_atom("true")), $3); }
| BOX ty                                                    { $$ = mk_node("TyBox", 1, $2); }
| '*' maybe_mut_or_const ty                                 { $$ = mk_node("TyPtr", 2, $2, $3); }
| '&' ty                                                    { $$ = mk_node("TyRptr", 2, mk_atom("MutImmutable"), $2); }
| '&' MUT ty                                                { $$ = mk_node("TyRptr", 2, mk_atom("MutMutable"), $3); }
| ANDAND ty                                                 { $$ = mk_node("TyRptr", 1, mk_node("TyRptr", 2, mk_atom("MutImmutable"), $2)); }
| ANDAND MUT ty                                             { $$ = mk_node("TyRptr", 1, mk_node("TyRptr", 2, mk_atom("MutMutable"), $3)); }
| '&' lifetime maybe_mut ty                                 { $$ = mk_node("TyRptr", 3, $2, $3, $4); }
| ANDAND lifetime maybe_mut ty                              { $$ = mk_node("TyRptr", 1, mk_node("TyRptr", 3, $2, $3, $4)); }
| '[' ty ']'                                                { $$ = mk_node("TyVec", 1, $2); }
| '[' ty ',' DOTDOT expr ']'                                { $$ = mk_node("TyFixedLengthVec", 2, $2, $5); }
| '[' ty ';' expr ']'                                       { $$ = mk_node("TyFixedLengthVec", 2, $2, $4); }
| TYPEOF '(' expr ')'                                       { $$ = mk_node("TyTypeof", 1, $3); }
| UNDERSCORE                                                { $$ = mk_atom("TyInfer"); }
| ty_bare_fn
| ty_proc
| for_in_type
;

ty_bare_fn
:                         FN ty_fn_decl { $$ = $2; }
| UNSAFE                  FN ty_fn_decl { $$ = $3; }
|        EXTERN maybe_abi FN ty_fn_decl { $$ = $4; }
| UNSAFE EXTERN maybe_abi FN ty_fn_decl { $$ = $5; }
;

ty_fn_decl
: generic_params fn_anon_params ret_ty { $$ = mk_node("TyFnDecl", 3, $1, $2, $3); }
;

ty_closure
: UNSAFE '|' anon_params '|' maybe_bounds ret_ty { $$ = mk_node("TyClosure", 3, $3, $5, $6); }
|        '|' anon_params '|' maybe_bounds ret_ty { $$ = mk_node("TyClosure", 3, $2, $4, $5); }
| UNSAFE OROR maybe_bounds ret_ty                { $$ = mk_node("TyClosure", 2, $3, $4); }
|        OROR maybe_bounds ret_ty                { $$ = mk_node("TyClosure", 2, $2, $3); }
;

ty_proc
: PROC generic_params fn_params maybe_bounds ret_ty { $$ = mk_node("TyProc", 4, $2, $3, $4, $5); }
;

for_in_type
: FOR '<' maybe_lifetimes '>' for_in_type_suffix { $$ = mk_node("ForInType", 2, $3, $5); }
;

for_in_type_suffix
: ty_proc
| ty_bare_fn
| trait_ref
| ty_closure
;

maybe_mut
: MUT              { $$ = mk_atom("MutMutable"); }
| %prec MUT %empty { $$ = mk_atom("MutImmutable"); }
;

maybe_mut_or_const
: MUT    { $$ = mk_atom("MutMutable"); }
| CONST  { $$ = mk_atom("MutImmutable"); }
| %empty { $$ = mk_atom("MutImmutable"); }
;

ty_qualified_path_and_generic_values
: ty_qualified_path maybe_bindings
{
  $$ = mk_node("GenericValues", 3, mk_none(), mk_node("TySums", 1, mk_node("TySum", 1, $1)), $2);
}
| ty_qualified_path ',' ty_sums maybe_bindings
{
  $$ = mk_node("GenericValues", 3, mk_none(), mk_node("TySums", 2, $1, $3), $4);
}
;

ty_qualified_path
: ty_sum AS trait_ref '>' MOD_SEP ident                     { $$ = mk_node("TyQualifiedPath", 3, $1, $3, $6); }
| ty_sum AS trait_ref '>' MOD_SEP ident '+' ty_param_bounds { $$ = mk_node("TyQualifiedPath", 3, $1, $3, $6); }
;

maybe_ty_sums
: ty_sums
| ty_sums ','
| %empty { $$ = mk_none(); }
;

ty_sums
: ty_sum             { $$ = mk_node("TySums", 1, $1); }
| ty_sums ',' ty_sum { $$ = ext_node($1, 1, $3); }
;

ty_sum
: ty                     { $$ = mk_node("TySum", 1, $1); }
| ty '+' ty_param_bounds { $$ = mk_node("TySum", 2, $1, $3); }
;

ty_prim_sum
: ty_prim                     { $$ = mk_node("TySum", 1, $1); }
| ty_prim '+' ty_param_bounds { $$ = mk_node("TySum", 2, $1, $3); }
;

maybe_ty_param_bounds
: ':' ty_param_bounds { $$ = $2; }
| %empty              { $$ = mk_none(); }
;

ty_param_bounds
: boundseq
| %empty { $$ = mk_none(); }
;

boundseq
: polybound
| boundseq '+' polybound { $$ = ext_node($1, 1, $3); }
;

polybound
: FOR '<' maybe_lifetimes '>' bound { $$ = mk_node("PolyBound", 2, $3, $5); }
| bound
| '?' bound { $$ = $2; }
;

bindings
: binding              { $$ = mk_node("Bindings", 1, $1); }
| bindings ',' binding { $$ = ext_node($1, 1, $3); }
;

binding
: ident '=' ty { mk_node("Binding", 2, $1, $3); }
;

ty_param
: ident maybe_ty_param_bounds maybe_ty_default           { $$ = mk_node("TyParam", 3, $1, $2, $3); }
| ident '?' ident maybe_ty_param_bounds maybe_ty_default { $$ = mk_node("TyParam", 4, $1, $3, $4, $5); }
;

maybe_bounds
: %prec SHIFTPLUS
  ':' bounds             { $$ = $2; }
| %prec SHIFTPLUS %empty { $$ = mk_none(); }
;

bounds
: bound            { $$ = mk_node("bounds", 1, $1); }
| bounds '+' bound { $$ = ext_node($1, 1, $3); }
;

bound
: lifetime
| trait_ref
;

maybe_ltbounds
: %prec SHIFTPLUS
  ':' ltbounds       { $$ = $2; }
| %empty             { $$ = mk_none(); }
;

ltbounds
: lifetime              { $$ = mk_node("ltbounds", 1, $1); }
| ltbounds '+' lifetime { $$ = ext_node($1, 1, $3); }
;

maybe_ty_default
: '=' ty_sum { $$ = mk_node("TyDefault", 1, $2); }
| %empty     { $$ = mk_none(); }
;

maybe_lifetimes
: lifetimes
| lifetimes ','
| %empty { $$ = mk_none(); }
;

lifetimes
: lifetime_and_bounds               { $$ = mk_node("Lifetimes", 1, $1); }
| lifetimes ',' lifetime_and_bounds { $$ = ext_node($1, 1, $3); }
;

lifetime_and_bounds
: LIFETIME maybe_ltbounds         { $$ = mk_node("lifetime", 2, mk_atom(yytext), $2); }
| STATIC_LIFETIME                 { $$ = mk_atom("static_lifetime"); }
;

lifetime
: LIFETIME         { $$ = mk_node("lifetime", 1, mk_atom(yytext)); }
| STATIC_LIFETIME  { $$ = mk_atom("static_lifetime"); }
;

trait_ref
: %prec IDENT path_generic_args_without_colons
| %prec IDENT MOD_SEP path_generic_args_without_colons { $$ = $2; }
;

////////////////////////////////////////////////////////////////////////
// Part 4: Blocks, statements, and expressions
////////////////////////////////////////////////////////////////////////

inner_attrs_and_block
: '{' maybe_inner_attrs maybe_stmts '}'        { $$ = mk_node("ExprBlock", 2, $2, $3); }
;

block
: '{' maybe_stmts '}'                          { $$ = mk_node("ExprBlock", 1, $2); }
;

maybe_stmts
: stmts
| stmts nonblock_expr { $$ = ext_node($1, 1, $2); }
| nonblock_expr
| %empty              { $$ = mk_none(); }
;

// There are two sub-grammars within a "stmts: exprs" derivation
// depending on whether each stmt-expr is a block-expr form; this is to
// handle the "semicolon rule" for stmt sequencing that permits
// writing
//
//     if foo { bar } 10
//
// as a sequence of two stmts (one if-expr stmt, one lit-10-expr
// stmt). Unfortunately by permitting juxtaposition of exprs in
// sequence like that, the non-block expr grammar has to have a
// second limited sub-grammar that excludes the prefix exprs that
// are ambiguous with binops. That is to say:
//
//     {10} - 1
//
// should parse as (progn (progn 10) (- 1)) not (- (progn 10) 1), that
// is to say, two statements rather than one, at least according to
// the mainline rust parser.
//
// So we wind up with a 3-way split in exprs that occur in stmt lists:
// block, nonblock-prefix, and nonblock-nonprefix.
//
// In non-stmts contexts, expr can relax this trichotomy.
//
// There is also one other expr subtype: nonparen_expr disallows exprs
// surrounded by parens (including tuple expressions), this is
// necessary for BOX (place) expressions, so a parens expr following
// the BOX is always parsed as the place.

stmts
: stmt           { $$ = mk_node("stmts", 1, $1); }
| stmts stmt     { $$ = ext_node($1, 1, $2); }
;

stmt
: let
|                 stmt_item
|             PUB stmt_item { $$ = $2; }
| outer_attrs     stmt_item { $$ = $2; }
| outer_attrs PUB stmt_item { $$ = $3; }
| full_block_expr
| block
| nonblock_expr ';'
| ';'                   { $$ = mk_none(); }
;

maybe_exprs
: exprs
| exprs ','
| %empty { $$ = mk_none(); }
;

maybe_expr
: expr
| %empty { $$ = mk_none(); }
;

exprs
: expr                                                        { $$ = mk_node("exprs", 1, $1); }
| exprs ',' expr                                              { $$ = ext_node($1, 1, $3); }
;

path_expr
: path_generic_args_with_colons
| MOD_SEP path_generic_args_with_colons      { $$ = $2; }
| SELF MOD_SEP path_generic_args_with_colons { $$ = mk_node("SelfPath", 1, $3); }
;

// A path with a lifetime and type parameters with double colons before
// the type parameters; e.g. `foo::bar::<'a>::Baz::<T>`
//
// These show up in expr context, in order to disambiguate from "less-than"
// expressions.
path_generic_args_with_colons
: ident                                              { $$ = mk_node("components", 1, $1); }
| path_generic_args_with_colons MOD_SEP ident        { $$ = ext_node($1, 1, $3); }
| path_generic_args_with_colons MOD_SEP generic_args { $$ = ext_node($1, 1, $3); }
;

// the braces-delimited macro is a block_expr so it doesn't appear here
macro_expr
: path_expr '!' maybe_ident parens_delimited_token_trees   { $$ = mk_node("MacroExpr", 3, $1, $3, $4); }
| path_expr '!' maybe_ident brackets_delimited_token_trees { $$ = mk_node("MacroExpr", 3, $1, $3, $4); }
;

nonblock_expr
: lit                                                           { $$ = mk_node("ExprLit", 1, $1); }
| %prec IDENT
  path_expr                                                     { $$ = mk_node("ExprPath", 1, $1); }
| SELF                                                          { $$ = mk_node("ExprPath", 1, mk_node("ident", 1, mk_atom("self"))); }
| macro_expr                                                    { $$ = mk_node("ExprMac", 1, $1); }
| path_expr '{' struct_expr_fields '}'                          { $$ = mk_node("ExprStruct", 2, $1, $3); }
| nonblock_expr '.' path_generic_args_with_colons               { $$ = mk_node("ExprField", 2, $1, $3); }
| nonblock_expr '.' LIT_INTEGER                                 { $$ = mk_node("ExprTupleIndex", 1, $1); }
| nonblock_expr '[' maybe_expr ']'                              { $$ = mk_node("ExprIndex", 2, $1, $3); }
| nonblock_expr '(' maybe_exprs ')'                             { $$ = mk_node("ExprCall", 2, $1, $3); }
| '[' vec_expr ']'                                              { $$ = mk_node("ExprVec", 1, $2); }
| '(' maybe_exprs ')'                                           { $$ = mk_node("ExprParen", 1, $2); }
| CONTINUE                                                      { $$ = mk_node("ExprAgain", 0); }
| CONTINUE lifetime                                             { $$ = mk_node("ExprAgain", 1, $2); }
| RETURN                                                        { $$ = mk_node("ExprRet", 0); }
| RETURN expr                                                   { $$ = mk_node("ExprRet", 1, $2); }
| BREAK                                                         { $$ = mk_node("ExprBreak", 0); }
| BREAK lifetime                                                { $$ = mk_node("ExprBreak", 1, $2); }
| nonblock_expr LARROW expr                                     { $$ = mk_node("ExprInPlace", 2, $1, $3); }
| nonblock_expr '=' expr                                        { $$ = mk_node("ExprAssign", 2, $1, $3); }
| nonblock_expr SHLEQ expr                                      { $$ = mk_node("ExprAssignShl", 2, $1, $3); }
| nonblock_expr SHREQ expr                                      { $$ = mk_node("ExprAssignShr", 2, $1, $3); }
| nonblock_expr MINUSEQ expr                                    { $$ = mk_node("ExprAssignSub", 2, $1, $3); }
| nonblock_expr ANDEQ expr                                      { $$ = mk_node("ExprAssignBitAnd", 2, $1, $3); }
| nonblock_expr OREQ expr                                       { $$ = mk_node("ExprAssignBitOr", 2, $1, $3); }
| nonblock_expr PLUSEQ expr                                     { $$ = mk_node("ExprAssignAdd", 2, $1, $3); }
| nonblock_expr STAREQ expr                                     { $$ = mk_node("ExprAssignMul", 2, $1, $3); }
| nonblock_expr SLASHEQ expr                                    { $$ = mk_node("ExprAssignDiv", 2, $1, $3); }
| nonblock_expr CARETEQ expr                                    { $$ = mk_node("ExprAssignBitXor", 2, $1, $3); }
| nonblock_expr PERCENTEQ expr                                  { $$ = mk_node("ExprAssignRem", 2, $1, $3); }
| nonblock_expr OROR expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiOr"), $1, $3); }
| nonblock_expr ANDAND expr                                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiAnd"), $1, $3); }
| nonblock_expr EQEQ expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiEq"), $1, $3); }
| nonblock_expr NE expr                                         { $$ = mk_node("ExprBinary", 3, mk_atom("BiNe"), $1, $3); }
| nonblock_expr '<' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiLt"), $1, $3); }
| nonblock_expr '>' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiGt"), $1, $3); }
| nonblock_expr LE expr                                         { $$ = mk_node("ExprBinary", 3, mk_atom("BiLe"), $1, $3); }
| nonblock_expr GE expr                                         { $$ = mk_node("ExprBinary", 3, mk_atom("BiGe"), $1, $3); }
| nonblock_expr '|' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitOr"), $1, $3); }
| nonblock_expr '^' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitXor"), $1, $3); }
| nonblock_expr '&' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitAnd"), $1, $3); }
| nonblock_expr SHL expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiShl"), $1, $3); }
| nonblock_expr SHR expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiShr"), $1, $3); }
| nonblock_expr '+' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiAdd"), $1, $3); }
| nonblock_expr '-' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiSub"), $1, $3); }
| nonblock_expr '*' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiMul"), $1, $3); }
| nonblock_expr '/' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiDiv"), $1, $3); }
| nonblock_expr '%' expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiRem"), $1, $3); }
| nonblock_expr DOTDOT                                          { $$ = mk_node("ExprRange", 2, $1, mk_none()); }
| nonblock_expr DOTDOT expr                                     { $$ = mk_node("ExprRange", 2, $1, $3); }
|               DOTDOT expr                                     { $$ = mk_node("ExprRange", 2, mk_none(), $2); }
|               DOTDOT                                          { $$ = mk_node("ExprRange", 2, mk_none(), mk_none()); }
| nonblock_expr AS ty                                           { $$ = mk_node("ExprCast", 2, $1, $3); }
| BOX nonparen_expr                                             { $$ = mk_node("ExprBox", 1, $2); }
| %prec BOXPLACE BOX '(' maybe_expr ')' nonblock_expr           { $$ = mk_node("ExprBox", 2, $3, $5); }
| expr_qualified_path
| nonblock_prefix_expr
;

expr
: lit                                                 { $$ = mk_node("ExprLit", 1, $1); }
| %prec IDENT
  path_expr                                           { $$ = mk_node("ExprPath", 1, $1); }
| SELF                                                { $$ = mk_node("ExprPath", 1, mk_node("ident", 1, mk_atom("self"))); }
| macro_expr                                          { $$ = mk_node("ExprMac", 1, $1); }
| path_expr '{' struct_expr_fields '}'                { $$ = mk_node("ExprStruct", 2, $1, $3); }
| expr '.' path_generic_args_with_colons              { $$ = mk_node("ExprField", 2, $1, $3); }
| expr '.' LIT_INTEGER                                { $$ = mk_node("ExprTupleIndex", 1, $1); }
| expr '[' maybe_expr ']'                             { $$ = mk_node("ExprIndex", 2, $1, $3); }
| expr '(' maybe_exprs ')'                            { $$ = mk_node("ExprCall", 2, $1, $3); }
| '(' maybe_exprs ')'                                 { $$ = mk_node("ExprParen", 1, $2); }
| '[' vec_expr ']'                                    { $$ = mk_node("ExprVec", 1, $2); }
| CONTINUE                                            { $$ = mk_node("ExprAgain", 0); }
| CONTINUE ident                                      { $$ = mk_node("ExprAgain", 1, $2); }
| RETURN                                              { $$ = mk_node("ExprRet", 0); }
| RETURN expr                                         { $$ = mk_node("ExprRet", 1, $2); }
| BREAK                                               { $$ = mk_node("ExprBreak", 0); }
| BREAK ident                                         { $$ = mk_node("ExprBreak", 1, $2); }
| expr LARROW expr                                    { $$ = mk_node("ExprInPlace", 2, $1, $3); }
| expr '=' expr                                       { $$ = mk_node("ExprAssign", 2, $1, $3); }
| expr SHLEQ expr                                     { $$ = mk_node("ExprAssignShl", 2, $1, $3); }
| expr SHREQ expr                                     { $$ = mk_node("ExprAssignShr", 2, $1, $3); }
| expr MINUSEQ expr                                   { $$ = mk_node("ExprAssignSub", 2, $1, $3); }
| expr ANDEQ expr                                     { $$ = mk_node("ExprAssignBitAnd", 2, $1, $3); }
| expr OREQ expr                                      { $$ = mk_node("ExprAssignBitOr", 2, $1, $3); }
| expr PLUSEQ expr                                    { $$ = mk_node("ExprAssignAdd", 2, $1, $3); }
| expr STAREQ expr                                    { $$ = mk_node("ExprAssignMul", 2, $1, $3); }
| expr SLASHEQ expr                                   { $$ = mk_node("ExprAssignDiv", 2, $1, $3); }
| expr CARETEQ expr                                   { $$ = mk_node("ExprAssignBitXor", 2, $1, $3); }
| expr PERCENTEQ expr                                 { $$ = mk_node("ExprAssignRem", 2, $1, $3); }
| expr OROR expr                                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiOr"), $1, $3); }
| expr ANDAND expr                                    { $$ = mk_node("ExprBinary", 3, mk_atom("BiAnd"), $1, $3); }
| expr EQEQ expr                                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiEq"), $1, $3); }
| expr NE expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiNe"), $1, $3); }
| expr '<' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiLt"), $1, $3); }
| expr '>' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiGt"), $1, $3); }
| expr LE expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiLe"), $1, $3); }
| expr GE expr                                        { $$ = mk_node("ExprBinary", 3, mk_atom("BiGe"), $1, $3); }
| expr '|' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitOr"), $1, $3); }
| expr '^' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitXor"), $1, $3); }
| expr '&' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitAnd"), $1, $3); }
| expr SHL expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiShl"), $1, $3); }
| expr SHR expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiShr"), $1, $3); }
| expr '+' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiAdd"), $1, $3); }
| expr '-' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiSub"), $1, $3); }
| expr '*' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiMul"), $1, $3); }
| expr '/' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiDiv"), $1, $3); }
| expr '%' expr                                       { $$ = mk_node("ExprBinary", 3, mk_atom("BiRem"), $1, $3); }
| expr DOTDOT                                         { $$ = mk_node("ExprRange", 2, $1, mk_none()); }
| expr DOTDOT expr                                    { $$ = mk_node("ExprRange", 2, $1, $3); }
|      DOTDOT expr                                    { $$ = mk_node("ExprRange", 2, mk_none(), $2); }
|      DOTDOT                                         { $$ = mk_node("ExprRange", 2, mk_none(), mk_none()); }
| expr AS ty                                          { $$ = mk_node("ExprCast", 2, $1, $3); }
| BOX nonparen_expr                                   { $$ = mk_node("ExprBox", 1, $2); }
| %prec BOXPLACE BOX '(' maybe_expr ')' expr          { $$ = mk_node("ExprBox", 2, $3, $5); }
| expr_qualified_path
| block_expr
| block
| nonblock_prefix_expr
;

nonparen_expr
: lit                                                 { $$ = mk_node("ExprLit", 1, $1); }
| %prec IDENT
  path_expr                                           { $$ = mk_node("ExprPath", 1, $1); }
| SELF                                                { $$ = mk_node("ExprPath", 1, mk_node("ident", 1, mk_atom("self"))); }
| macro_expr                                          { $$ = mk_node("ExprMac", 1, $1); }
| path_expr '{' struct_expr_fields '}'                { $$ = mk_node("ExprStruct", 2, $1, $3); }
| nonparen_expr '.' path_generic_args_with_colons     { $$ = mk_node("ExprField", 2, $1, $3); }
| nonparen_expr '.' LIT_INTEGER                       { $$ = mk_node("ExprTupleIndex", 1, $1); }
| nonparen_expr '[' maybe_expr ']'                    { $$ = mk_node("ExprIndex", 2, $1, $3); }
| nonparen_expr '(' maybe_exprs ')'                   { $$ = mk_node("ExprCall", 2, $1, $3); }
| '[' vec_expr ']'                                    { $$ = mk_node("ExprVec", 1, $2); }
| CONTINUE                                            { $$ = mk_node("ExprAgain", 0); }
| CONTINUE ident                                      { $$ = mk_node("ExprAgain", 1, $2); }
| RETURN                                              { $$ = mk_node("ExprRet", 0); }
| RETURN expr                                         { $$ = mk_node("ExprRet", 1, $2); }
| BREAK                                               { $$ = mk_node("ExprBreak", 0); }
| BREAK ident                                         { $$ = mk_node("ExprBreak", 1, $2); }
| nonparen_expr LARROW nonparen_expr                  { $$ = mk_node("ExprInPlace", 2, $1, $3); }
| nonparen_expr '=' nonparen_expr                     { $$ = mk_node("ExprAssign", 2, $1, $3); }
| nonparen_expr SHLEQ nonparen_expr                   { $$ = mk_node("ExprAssignShl", 2, $1, $3); }
| nonparen_expr SHREQ nonparen_expr                   { $$ = mk_node("ExprAssignShr", 2, $1, $3); }
| nonparen_expr MINUSEQ nonparen_expr                 { $$ = mk_node("ExprAssignSub", 2, $1, $3); }
| nonparen_expr ANDEQ nonparen_expr                   { $$ = mk_node("ExprAssignBitAnd", 2, $1, $3); }
| nonparen_expr OREQ nonparen_expr                    { $$ = mk_node("ExprAssignBitOr", 2, $1, $3); }
| nonparen_expr PLUSEQ nonparen_expr                  { $$ = mk_node("ExprAssignAdd", 2, $1, $3); }
| nonparen_expr STAREQ nonparen_expr                  { $$ = mk_node("ExprAssignMul", 2, $1, $3); }
| nonparen_expr SLASHEQ nonparen_expr                 { $$ = mk_node("ExprAssignDiv", 2, $1, $3); }
| nonparen_expr CARETEQ nonparen_expr                 { $$ = mk_node("ExprAssignBitXor", 2, $1, $3); }
| nonparen_expr PERCENTEQ nonparen_expr               { $$ = mk_node("ExprAssignRem", 2, $1, $3); }
| nonparen_expr OROR nonparen_expr                    { $$ = mk_node("ExprBinary", 3, mk_atom("BiOr"), $1, $3); }
| nonparen_expr ANDAND nonparen_expr                  { $$ = mk_node("ExprBinary", 3, mk_atom("BiAnd"), $1, $3); }
| nonparen_expr EQEQ nonparen_expr                    { $$ = mk_node("ExprBinary", 3, mk_atom("BiEq"), $1, $3); }
| nonparen_expr NE nonparen_expr                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiNe"), $1, $3); }
| nonparen_expr '<' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiLt"), $1, $3); }
| nonparen_expr '>' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiGt"), $1, $3); }
| nonparen_expr LE nonparen_expr                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiLe"), $1, $3); }
| nonparen_expr GE nonparen_expr                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiGe"), $1, $3); }
| nonparen_expr '|' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitOr"), $1, $3); }
| nonparen_expr '^' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitXor"), $1, $3); }
| nonparen_expr '&' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitAnd"), $1, $3); }
| nonparen_expr SHL nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiShl"), $1, $3); }
| nonparen_expr SHR nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiShr"), $1, $3); }
| nonparen_expr '+' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiAdd"), $1, $3); }
| nonparen_expr '-' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiSub"), $1, $3); }
| nonparen_expr '*' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiMul"), $1, $3); }
| nonparen_expr '/' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiDiv"), $1, $3); }
| nonparen_expr '%' nonparen_expr                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiRem"), $1, $3); }
| nonparen_expr DOTDOT                                { $$ = mk_node("ExprRange", 2, $1, mk_none()); }
| nonparen_expr DOTDOT nonparen_expr                  { $$ = mk_node("ExprRange", 2, $1, $3); }
|               DOTDOT nonparen_expr                  { $$ = mk_node("ExprRange", 2, mk_none(), $2); }
|               DOTDOT                                { $$ = mk_node("ExprRange", 2, mk_none(), mk_none()); }
| nonparen_expr AS ty                                 { $$ = mk_node("ExprCast", 2, $1, $3); }
| BOX nonparen_expr                                   { $$ = mk_node("ExprBox", 1, $2); }
| %prec BOXPLACE BOX '(' maybe_expr ')' expr          { $$ = mk_node("ExprBox", 1, $3, $5); }
| expr_qualified_path
| block_expr
| block
| nonblock_prefix_expr
;

expr_nostruct
: lit                                                 { $$ = mk_node("ExprLit", 1, $1); }
| %prec IDENT
  path_expr                                           { $$ = mk_node("ExprPath", 1, $1); }
| SELF                                                { $$ = mk_node("ExprPath", 1, mk_node("ident", 1, mk_atom("self"))); }
| macro_expr                                          { $$ = mk_node("ExprMac", 1, $1); }
| expr_nostruct '.' path_generic_args_with_colons     { $$ = mk_node("ExprField", 2, $1, $3); }
| expr_nostruct '.' LIT_INTEGER                       { $$ = mk_node("ExprTupleIndex", 1, $1); }
| expr_nostruct '[' maybe_expr ']'                    { $$ = mk_node("ExprIndex", 2, $1, $3); }
| expr_nostruct '(' maybe_exprs ')'                   { $$ = mk_node("ExprCall", 2, $1, $3); }
| '[' vec_expr ']'                                    { $$ = mk_node("ExprVec", 1, $2); }
| '(' maybe_exprs ')'                                 { $$ = mk_node("ExprParen", 1, $2); }
| CONTINUE                                            { $$ = mk_node("ExprAgain", 0); }
| CONTINUE ident                                      { $$ = mk_node("ExprAgain", 1, $2); }
| RETURN                                              { $$ = mk_node("ExprRet", 0); }
| RETURN expr                                         { $$ = mk_node("ExprRet", 1, $2); }
| BREAK                                               { $$ = mk_node("ExprBreak", 0); }
| BREAK ident                                         { $$ = mk_node("ExprBreak", 1, $2); }
| expr_nostruct LARROW expr_nostruct                  { $$ = mk_node("ExprInPlace", 2, $1, $3); }
| expr_nostruct '=' expr_nostruct                     { $$ = mk_node("ExprAssign", 2, $1, $3); }
| expr_nostruct SHLEQ expr_nostruct                   { $$ = mk_node("ExprAssignShl", 2, $1, $3); }
| expr_nostruct SHREQ expr_nostruct                   { $$ = mk_node("ExprAssignShr", 2, $1, $3); }
| expr_nostruct MINUSEQ expr_nostruct                 { $$ = mk_node("ExprAssignSub", 2, $1, $3); }
| expr_nostruct ANDEQ expr_nostruct                   { $$ = mk_node("ExprAssignBitAnd", 2, $1, $3); }
| expr_nostruct OREQ expr_nostruct                    { $$ = mk_node("ExprAssignBitOr", 2, $1, $3); }
| expr_nostruct PLUSEQ expr_nostruct                  { $$ = mk_node("ExprAssignAdd", 2, $1, $3); }
| expr_nostruct STAREQ expr_nostruct                  { $$ = mk_node("ExprAssignMul", 2, $1, $3); }
| expr_nostruct SLASHEQ expr_nostruct                 { $$ = mk_node("ExprAssignDiv", 2, $1, $3); }
| expr_nostruct CARETEQ expr_nostruct                 { $$ = mk_node("ExprAssignBitXor", 2, $1, $3); }
| expr_nostruct PERCENTEQ expr_nostruct               { $$ = mk_node("ExprAssignRem", 2, $1, $3); }
| expr_nostruct OROR expr_nostruct                    { $$ = mk_node("ExprBinary", 3, mk_atom("BiOr"), $1, $3); }
| expr_nostruct ANDAND expr_nostruct                  { $$ = mk_node("ExprBinary", 3, mk_atom("BiAnd"), $1, $3); }
| expr_nostruct EQEQ expr_nostruct                    { $$ = mk_node("ExprBinary", 3, mk_atom("BiEq"), $1, $3); }
| expr_nostruct NE expr_nostruct                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiNe"), $1, $3); }
| expr_nostruct '<' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiLt"), $1, $3); }
| expr_nostruct '>' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiGt"), $1, $3); }
| expr_nostruct LE expr_nostruct                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiLe"), $1, $3); }
| expr_nostruct GE expr_nostruct                      { $$ = mk_node("ExprBinary", 3, mk_atom("BiGe"), $1, $3); }
| expr_nostruct '|' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitOr"), $1, $3); }
| expr_nostruct '^' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitXor"), $1, $3); }
| expr_nostruct '&' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiBitAnd"), $1, $3); }
| expr_nostruct SHL expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiShl"), $1, $3); }
| expr_nostruct SHR expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiShr"), $1, $3); }
| expr_nostruct '+' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiAdd"), $1, $3); }
| expr_nostruct '-' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiSub"), $1, $3); }
| expr_nostruct '*' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiMul"), $1, $3); }
| expr_nostruct '/' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiDiv"), $1, $3); }
| expr_nostruct '%' expr_nostruct                     { $$ = mk_node("ExprBinary", 3, mk_atom("BiRem"), $1, $3); }
| expr_nostruct DOTDOT               %prec RANGE      { $$ = mk_node("ExprRange", 2, $1, mk_none()); }
| expr_nostruct DOTDOT expr_nostruct                  { $$ = mk_node("ExprRange", 2, $1, $3); }
|               DOTDOT expr_nostruct                  { $$ = mk_node("ExprRange", 2, mk_none(), $2); }
|               DOTDOT                                { $$ = mk_node("ExprRange", 2, mk_none(), mk_none()); }
| expr_nostruct AS ty                                 { $$ = mk_node("ExprCast", 2, $1, $3); }
| BOX nonparen_expr                                   { $$ = mk_node("ExprBox", 1, $2); }
| %prec BOXPLACE BOX '(' maybe_expr ')' expr_nostruct { $$ = mk_node("ExprBox", 1, $3, $5); }
| expr_qualified_path
| block_expr
| block
| nonblock_prefix_expr_nostruct
;

nonblock_prefix_expr_nostruct
: '-' expr_nostruct                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnNeg"), $2); }
| '!' expr_nostruct                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnNot"), $2); }
| '*' expr_nostruct                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnDeref"), $2); }
| '&' maybe_mut expr_nostruct               { $$ = mk_node("ExprAddrOf", 2, $2, $3); }
| ANDAND maybe_mut expr_nostruct            { $$ = mk_node("ExprAddrOf", 1, mk_node("ExprAddrOf", 2, $2, $3)); }
| lambda_expr_nostruct
| MOVE lambda_expr_nostruct                 { $$ = $2; }
| proc_expr_nostruct
;

nonblock_prefix_expr
: '-' expr                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnNeg"), $2); }
| '!' expr                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnNot"), $2); }
| '*' expr                         { $$ = mk_node("ExprUnary", 2, mk_atom("UnDeref"), $2); }
| '&' maybe_mut expr               { $$ = mk_node("ExprAddrOf", 2, $2, $3); }
| ANDAND maybe_mut expr            { $$ = mk_node("ExprAddrOf", 1, mk_node("ExprAddrOf", 2, $2, $3)); }
| lambda_expr
| MOVE lambda_expr                 { $$ = $2; }
| proc_expr
;

expr_qualified_path
: '<' ty_sum maybe_as_trait_ref '>' MOD_SEP ident maybe_qpath_params
{
  $$ = mk_node("ExprQualifiedPath", 4, $2, $3, $6, $7);
}
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident maybe_as_trait_ref '>' MOD_SEP ident
{
  $$ = mk_node("ExprQualifiedPath", 3, mk_node("ExprQualifiedPath", 3, $2, $3, $6), $7, $10);
}
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident generic_args maybe_as_trait_ref '>' MOD_SEP ident
{
  $$ = mk_node("ExprQualifiedPath", 3, mk_node("ExprQualifiedPath", 4, $2, $3, $6, $7), $8, $11);
}
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident maybe_as_trait_ref '>' MOD_SEP ident generic_args
{
  $$ = mk_node("ExprQualifiedPath", 4, mk_node("ExprQualifiedPath", 3, $2, $3, $6), $7, $10, $11);
}
| SHL ty_sum maybe_as_trait_ref '>' MOD_SEP ident generic_args maybe_as_trait_ref '>' MOD_SEP ident generic_args
{
  $$ = mk_node("ExprQualifiedPath", 4, mk_node("ExprQualifiedPath", 4, $2, $3, $6, $7), $8, $11, $12);
}

maybe_qpath_params
: MOD_SEP generic_args { $$ = $2; }
| %empty               { $$ = mk_none(); }
;

maybe_as_trait_ref
: AS trait_ref { $$ = $2; }
| %empty       { $$ = mk_none(); }
;

lambda_expr
: %prec LAMBDA
  OROR ret_ty expr                                        { $$ = mk_node("ExprFnBlock", 3, mk_none(), $2, $3); }
| %prec LAMBDA
  '|' maybe_unboxed_closure_kind '|' ret_ty expr          { $$ = mk_node("ExprFnBlock", 3, mk_none(), $4, $5); }
| %prec LAMBDA
  '|' inferrable_params '|' ret_ty expr                   { $$ = mk_node("ExprFnBlock", 3, $2, $4, $5); }
| %prec LAMBDA
  '|' '&' maybe_mut ':' inferrable_params '|' ret_ty expr { $$ = mk_node("ExprFnBlock", 3, $5, $7, $8); }
| %prec LAMBDA
  '|' ':' inferrable_params '|' ret_ty expr               { $$ = mk_node("ExprFnBlock", 3, $3, $5, $6); }
;

lambda_expr_nostruct
: %prec LAMBDA
  OROR expr_nostruct                                        { $$ = mk_node("ExprFnBlock", 2, mk_none(), $2); }
| %prec LAMBDA
  '|' maybe_unboxed_closure_kind '|'  expr_nostruct         { $$ = mk_node("ExprFnBlock", 2, mk_none(), $4); }
| %prec LAMBDA
  '|' inferrable_params '|' expr_nostruct                   { $$ = mk_node("ExprFnBlock", 2, $2, $4); }
| %prec LAMBDA
  '|' '&' maybe_mut ':' inferrable_params '|' expr_nostruct { $$ = mk_node("ExprFnBlock", 2, $5, $7); }
| %prec LAMBDA
  '|' ':' inferrable_params '|' expr_nostruct               { $$ = mk_node("ExprFnBlock", 2, $3, $5); }

;

proc_expr
: %prec LAMBDA
  PROC '(' ')' expr                         { $$ = mk_node("ExprProc", 2, mk_none(), $4); }
| %prec LAMBDA
  PROC '(' inferrable_params ')' expr       { $$ = mk_node("ExprProc", 2, $3, $5); }
;

proc_expr_nostruct
: %prec LAMBDA
  PROC '(' ')' expr_nostruct                     { $$ = mk_node("ExprProc", 2, mk_none(), $4); }
| %prec LAMBDA
  PROC '(' inferrable_params ')' expr_nostruct   { $$ = mk_node("ExprProc", 2, $3, $5); }
;

vec_expr
: maybe_exprs
| exprs ';' expr { $$ = mk_node("VecRepeat", 2, $1, $3); }
;

struct_expr_fields
: field_inits
| field_inits ','
| maybe_field_inits default_field_init { $$ = ext_node($1, 1, $2); }
;

maybe_field_inits
: field_inits
| field_inits ','
| %empty { $$ = mk_none(); }
;

field_inits
: field_init                 { $$ = mk_node("FieldInits", 1, $1); }
| field_inits ',' field_init { $$ = ext_node($1, 1, $3); }
;

field_init
: ident ':' expr   { $$ = mk_node("FieldInit", 2, $1, $3); }
;

default_field_init
: DOTDOT expr   { $$ = mk_node("DefaultFieldInit", 1, $2); }
;

block_expr
: expr_match
| expr_if
| expr_if_let
| expr_while
| expr_while_let
| expr_loop
| expr_for
| UNSAFE block                                           { $$ = mk_node("UnsafeBlock", 1, $2); }
| path_expr '!' maybe_ident braces_delimited_token_trees { $$ = mk_node("Macro", 3, $1, $3, $4); }
;

full_block_expr
: block_expr
| full_block_expr '.' path_generic_args_with_colons %prec IDENT         { $$ = mk_node("ExprField", 2, $1, $3); }
| full_block_expr '.' path_generic_args_with_colons '[' maybe_expr ']'  { $$ = mk_node("ExprIndex", 3, $1, $3, $5); }
| full_block_expr '.' path_generic_args_with_colons '(' maybe_exprs ')' { $$ = mk_node("ExprCall", 3, $1, $3, $5); }
| full_block_expr '.' LIT_INTEGER                                       { $$ = mk_node("ExprTupleIndex", 1, $1); }
;

expr_match
: MATCH expr_nostruct '{' '}'                                     { $$ = mk_node("ExprMatch", 1, $2); }
| MATCH expr_nostruct '{' match_clauses                       '}' { $$ = mk_node("ExprMatch", 2, $2, $4); }
| MATCH expr_nostruct '{' match_clauses nonblock_match_clause '}' { $$ = mk_node("ExprMatch", 2, $2, ext_node($4, 1, $5)); }
| MATCH expr_nostruct '{'               nonblock_match_clause '}' { $$ = mk_node("ExprMatch", 2, $2, mk_node("Arms", 1, $4)); }
;

match_clauses
: match_clause               { $$ = mk_node("Arms", 1, $1); }
| match_clauses match_clause { $$ = ext_node($1, 1, $2); }
;

match_clause
: nonblock_match_clause ','
| block_match_clause
| block_match_clause ','
;

nonblock_match_clause
: maybe_outer_attrs pats_or maybe_guard FAT_ARROW nonblock_expr   { $$ = mk_node("Arm", 4, $1, $2, $3, $5); }
| maybe_outer_attrs pats_or maybe_guard FAT_ARROW full_block_expr { $$ = mk_node("Arm", 4, $1, $2, $3, $5); }
;

block_match_clause
: maybe_outer_attrs pats_or maybe_guard FAT_ARROW block { $$ = mk_node("Arm", 4, $1, $2, $3, $5); }
;

maybe_guard
: IF expr_nostruct           { $$ = $2; }
| %empty                     { $$ = mk_none(); }
;

expr_if
: IF expr_nostruct block                              { $$ = mk_node("ExprIf", 2, $2, $3); }
| IF expr_nostruct block ELSE block_or_if             { $$ = mk_node("ExprIf", 3, $2, $3, $5); }
;

expr_if_let
: IF LET pat '=' expr_nostruct block                  { $$ = mk_node("ExprIfLet", 3, $3, $5, $6); }
| IF LET pat '=' expr_nostruct block ELSE block_or_if { $$ = mk_node("ExprIfLet", 4, $3, $5, $6, $8); }
;

block_or_if
: block
| expr_if
| expr_if_let
;

expr_while
: maybe_label WHILE expr_nostruct block               { $$ = mk_node("ExprWhile", 3, $1, $3, $4); }
;

expr_while_let
: maybe_label WHILE LET pat '=' expr_nostruct block   { $$ = mk_node("ExprWhileLet", 4, $1, $4, $6, $7); }
;

expr_loop
: maybe_label LOOP block                              { $$ = mk_node("ExprLoop", 2, $1, $3); }
;

expr_for
: maybe_label FOR pat IN expr_nostruct block          { $$ = mk_node("ExprForLoop", 4, $1, $3, $5, $6); }
;

maybe_label
: lifetime ':'
| %empty { $$ = mk_none(); }
;

let
: LET pat maybe_ty_ascription maybe_init_expr ';' { $$ = mk_node("DeclLocal", 3, $2, $3, $4); }
;

////////////////////////////////////////////////////////////////////////
// Part 5: Macros and misc. rules
////////////////////////////////////////////////////////////////////////

lit
: LIT_BYTE                   { $$ = mk_node("LitByte", 1, mk_atom(yytext)); }
| LIT_CHAR                   { $$ = mk_node("LitChar", 1, mk_atom(yytext)); }
| LIT_INTEGER                { $$ = mk_node("LitInteger", 1, mk_atom(yytext)); }
| LIT_FLOAT                  { $$ = mk_node("LitFloat", 1, mk_atom(yytext)); }
| TRUE                       { $$ = mk_node("LitBool", 1, mk_atom(yytext)); }
| FALSE                      { $$ = mk_node("LitBool", 1, mk_atom(yytext)); }
| str
;

str
: LIT_STR                    { $$ = mk_node("LitStr", 1, mk_atom(yytext), mk_atom("CookedStr")); }
| LIT_STR_RAW                { $$ = mk_node("LitStr", 1, mk_atom(yytext), mk_atom("RawStr")); }
| LIT_BYTE_STR                 { $$ = mk_node("LitByteStr", 1, mk_atom(yytext), mk_atom("ByteStr")); }
| LIT_BYTE_STR_RAW             { $$ = mk_node("LitByteStr", 1, mk_atom(yytext), mk_atom("RawByteStr")); }
;

maybe_ident
: %empty { $$ = mk_none(); }
| ident
;

ident
: IDENT                      { $$ = mk_node("ident", 1, mk_atom(yytext)); }
;

unpaired_token
: SHL                        { $$ = mk_atom(yytext); }
| SHR                        { $$ = mk_atom(yytext); }
| LE                         { $$ = mk_atom(yytext); }
| EQEQ                       { $$ = mk_atom(yytext); }
| NE                         { $$ = mk_atom(yytext); }
| GE                         { $$ = mk_atom(yytext); }
| ANDAND                     { $$ = mk_atom(yytext); }
| OROR                       { $$ = mk_atom(yytext); }
| LARROW                     { $$ = mk_atom(yytext); }
| SHLEQ                      { $$ = mk_atom(yytext); }
| SHREQ                      { $$ = mk_atom(yytext); }
| MINUSEQ                    { $$ = mk_atom(yytext); }
| ANDEQ                      { $$ = mk_atom(yytext); }
| OREQ                       { $$ = mk_atom(yytext); }
| PLUSEQ                     { $$ = mk_atom(yytext); }
| STAREQ                     { $$ = mk_atom(yytext); }
| SLASHEQ                    { $$ = mk_atom(yytext); }
| CARETEQ                    { $$ = mk_atom(yytext); }
| PERCENTEQ                  { $$ = mk_atom(yytext); }
| DOTDOT                     { $$ = mk_atom(yytext); }
| DOTDOTDOT                  { $$ = mk_atom(yytext); }
| MOD_SEP                    { $$ = mk_atom(yytext); }
| RARROW                     { $$ = mk_atom(yytext); }
| FAT_ARROW                  { $$ = mk_atom(yytext); }
| LIT_BYTE                   { $$ = mk_atom(yytext); }
| LIT_CHAR                   { $$ = mk_atom(yytext); }
| LIT_INTEGER                { $$ = mk_atom(yytext); }
| LIT_FLOAT                  { $$ = mk_atom(yytext); }
| LIT_STR                    { $$ = mk_atom(yytext); }
| LIT_STR_RAW                { $$ = mk_atom(yytext); }
| LIT_BYTE_STR                 { $$ = mk_atom(yytext); }
| LIT_BYTE_STR_RAW             { $$ = mk_atom(yytext); }
| IDENT                      { $$ = mk_atom(yytext); }
| UNDERSCORE                 { $$ = mk_atom(yytext); }
| LIFETIME                   { $$ = mk_atom(yytext); }
| SELF                       { $$ = mk_atom(yytext); }
| STATIC                     { $$ = mk_atom(yytext); }
| AS                         { $$ = mk_atom(yytext); }
| BREAK                      { $$ = mk_atom(yytext); }
| CRATE                      { $$ = mk_atom(yytext); }
| ELSE                       { $$ = mk_atom(yytext); }
| ENUM                       { $$ = mk_atom(yytext); }
| EXTERN                     { $$ = mk_atom(yytext); }
| FALSE                      { $$ = mk_atom(yytext); }
| FN                         { $$ = mk_atom(yytext); }
| FOR                        { $$ = mk_atom(yytext); }
| IF                         { $$ = mk_atom(yytext); }
| IMPL                       { $$ = mk_atom(yytext); }
| IN                         { $$ = mk_atom(yytext); }
| LET                        { $$ = mk_atom(yytext); }
| LOOP                       { $$ = mk_atom(yytext); }
| MATCH                      { $$ = mk_atom(yytext); }
| MOD                        { $$ = mk_atom(yytext); }
| MOVE                       { $$ = mk_atom(yytext); }
| MUT                        { $$ = mk_atom(yytext); }
| PRIV                       { $$ = mk_atom(yytext); }
| PUB                        { $$ = mk_atom(yytext); }
| REF                        { $$ = mk_atom(yytext); }
| RETURN                     { $$ = mk_atom(yytext); }
| STRUCT                     { $$ = mk_atom(yytext); }
| TRUE                       { $$ = mk_atom(yytext); }
| TRAIT                      { $$ = mk_atom(yytext); }
| TYPE                       { $$ = mk_atom(yytext); }
| UNSAFE                     { $$ = mk_atom(yytext); }
| USE                        { $$ = mk_atom(yytext); }
| WHILE                      { $$ = mk_atom(yytext); }
| CONTINUE                   { $$ = mk_atom(yytext); }
| PROC                       { $$ = mk_atom(yytext); }
| BOX                        { $$ = mk_atom(yytext); }
| CONST                      { $$ = mk_atom(yytext); }
| WHERE                      { $$ = mk_atom(yytext); }
| TYPEOF                     { $$ = mk_atom(yytext); }
| INNER_DOC_COMMENT          { $$ = mk_atom(yytext); }
| OUTER_DOC_COMMENT          { $$ = mk_atom(yytext); }
| SHEBANG                    { $$ = mk_atom(yytext); }
| STATIC_LIFETIME            { $$ = mk_atom(yytext); }
| ';'                        { $$ = mk_atom(yytext); }
| ','                        { $$ = mk_atom(yytext); }
| '.'                        { $$ = mk_atom(yytext); }
| '@'                        { $$ = mk_atom(yytext); }
| '#'                        { $$ = mk_atom(yytext); }
| '~'                        { $$ = mk_atom(yytext); }
| ':'                        { $$ = mk_atom(yytext); }
| '$'                        { $$ = mk_atom(yytext); }
| '='                        { $$ = mk_atom(yytext); }
| '?'                        { $$ = mk_atom(yytext); }
| '!'                        { $$ = mk_atom(yytext); }
| '<'                        { $$ = mk_atom(yytext); }
| '>'                        { $$ = mk_atom(yytext); }
| '-'                        { $$ = mk_atom(yytext); }
| '&'                        { $$ = mk_atom(yytext); }
| '|'                        { $$ = mk_atom(yytext); }
| '+'                        { $$ = mk_atom(yytext); }
| '*'                        { $$ = mk_atom(yytext); }
| '/'                        { $$ = mk_atom(yytext); }
| '^'                        { $$ = mk_atom(yytext); }
| '%'                        { $$ = mk_atom(yytext); }
;

token_trees
: %empty                     { $$ = mk_node("TokenTrees", 0); }
| token_trees token_tree     { $$ = ext_node($1, 1, $2); }
;

token_tree
: delimited_token_trees
| unpaired_token         { $$ = mk_node("TTTok", 1, $1); }
;

delimited_token_trees
: parens_delimited_token_trees
| braces_delimited_token_trees
| brackets_delimited_token_trees
;

parens_delimited_token_trees
: '(' token_trees ')'
{
  $$ = mk_node("TTDelim", 3,
               mk_node("TTTok", 1, mk_atom("(")),
               $2,
               mk_node("TTTok", 1, mk_atom(")")));
}
;

braces_delimited_token_trees
: '{' token_trees '}'
{
  $$ = mk_node("TTDelim", 3,
               mk_node("TTTok", 1, mk_atom("{")),
               $2,
               mk_node("TTTok", 1, mk_atom("}")));
}
;

brackets_delimited_token_trees
: '[' token_trees ']'
{
  $$ = mk_node("TTDelim", 3,
               mk_node("TTTok", 1, mk_atom("[")),
               $2,
               mk_node("TTTok", 1, mk_atom("]")));
}
;
