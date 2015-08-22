#![feature(plugin_registrar, box_syntax)]
#![feature(rustc_private, core, collections)]
#![feature(str_split_at, num_bits_bytes)]
#![allow(unknown_lints)]

#[macro_use]
extern crate syntax;
#[macro_use]
extern crate rustc;

// Only for the compile time checking of paths
extern crate core;
extern crate collections;

use rustc::plugin::Registry;
use rustc::lint::LintPassObject;

#[macro_use]
pub mod utils;
pub mod consts;
pub mod types;
pub mod misc;
pub mod eq_op;
pub mod bit_mask;
pub mod ptr_arg;
pub mod needless_bool;
pub mod approx_const;
pub mod eta_reduction;
pub mod identity_op;
pub mod mut_mut;
pub mod len_zero;
pub mod attrs;
pub mod collapsible_if;
pub mod unicode;
pub mod strings;
pub mod methods;
pub mod returns;
pub mod lifetimes;
pub mod loops;
pub mod ranges;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_lint_pass(box types::TypePass as LintPassObject);
    reg.register_lint_pass(box misc::MiscPass as LintPassObject);
    reg.register_lint_pass(box misc::TopLevelRefPass as LintPassObject);
    reg.register_lint_pass(box misc::CmpNan as LintPassObject);
    reg.register_lint_pass(box eq_op::EqOp as LintPassObject);
    reg.register_lint_pass(box bit_mask::BitMask as LintPassObject);
    reg.register_lint_pass(box ptr_arg::PtrArg as LintPassObject);
    reg.register_lint_pass(box needless_bool::NeedlessBool as LintPassObject);
    reg.register_lint_pass(box approx_const::ApproxConstant as LintPassObject);
    reg.register_lint_pass(box misc::FloatCmp as LintPassObject);
    reg.register_lint_pass(box misc::Precedence as LintPassObject);
    reg.register_lint_pass(box eta_reduction::EtaPass as LintPassObject);
    reg.register_lint_pass(box identity_op::IdentityOp as LintPassObject);
    reg.register_lint_pass(box mut_mut::MutMut as LintPassObject);
    reg.register_lint_pass(box len_zero::LenZero as LintPassObject);
    reg.register_lint_pass(box misc::CmpOwned as LintPassObject);
    reg.register_lint_pass(box attrs::AttrPass as LintPassObject);
    reg.register_lint_pass(box collapsible_if::CollapsibleIf as LintPassObject);
    reg.register_lint_pass(box misc::ModuloOne as LintPassObject);
    reg.register_lint_pass(box unicode::Unicode as LintPassObject);
    reg.register_lint_pass(box strings::StringAdd as LintPassObject);
    reg.register_lint_pass(box returns::ReturnPass as LintPassObject);
    reg.register_lint_pass(box methods::MethodsPass as LintPassObject);
    reg.register_lint_pass(box types::LetPass as LintPassObject);
    reg.register_lint_pass(box types::UnitCmp as LintPassObject);
    reg.register_lint_pass(box loops::LoopsPass as LintPassObject);
    reg.register_lint_pass(box lifetimes::LifetimePass as LintPassObject);
    reg.register_lint_pass(box ranges::StepByZero as LintPassObject);
    reg.register_lint_pass(box types::CastPass as LintPassObject);
    reg.register_lint_pass(box types::TypeComplexityPass as LintPassObject);

    reg.register_lint_group("clippy", vec![
        approx_const::APPROX_CONSTANT,
        attrs::INLINE_ALWAYS,
        bit_mask::BAD_BIT_MASK,
        bit_mask::INEFFECTIVE_BIT_MASK,
        collapsible_if::COLLAPSIBLE_IF,
        eq_op::EQ_OP,
        eta_reduction::REDUNDANT_CLOSURE,
        identity_op::IDENTITY_OP,
        len_zero::LEN_WITHOUT_IS_EMPTY,
        len_zero::LEN_ZERO,
        lifetimes::NEEDLESS_LIFETIMES,
        loops::EXPLICIT_ITER_LOOP,
        loops::ITER_NEXT_LOOP,
        loops::NEEDLESS_RANGE_LOOP,
        methods::OPTION_UNWRAP_USED,
        methods::RESULT_UNWRAP_USED,
        methods::STR_TO_STRING,
        methods::STRING_TO_STRING,
        misc::CMP_NAN,
        misc::CMP_OWNED,
        misc::FLOAT_CMP,
        misc::MODULO_ONE,
        misc::PRECEDENCE,
        misc::SINGLE_MATCH,
        misc::TOPLEVEL_REF_ARG,
        mut_mut::MUT_MUT,
        needless_bool::NEEDLESS_BOOL,
        ptr_arg::PTR_ARG,
        ranges::RANGE_STEP_BY_ZERO,
        returns::LET_AND_RETURN,
        returns::NEEDLESS_RETURN,
        strings::STRING_ADD,
        strings::STRING_ADD_ASSIGN,
        types::BOX_VEC,
        types::CAST_POSSIBLE_TRUNCATION,
        types::CAST_PRECISION_LOSS,
        types::CAST_SIGN_LOSS,
        types::LET_UNIT_VALUE,
        types::LINKEDLIST,
        types::TYPE_COMPLEXITY,
        types::UNIT_CMP,
        unicode::NON_ASCII_LITERAL,
        unicode::ZERO_WIDTH_SPACE,
    ]);
}
