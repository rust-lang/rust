export as_prec;
export unop_prec;
export binop_prec_table;
export op_spec;

#[doc = "Unary operators have higher precedence than binary"]
const unop_prec: int = 100;

#[doc = "
Precedence of the `as` operator, which is a binary operator
but is not represented in the precedence table.
"]
const as_prec: int = 11;

type op_spec = {tok: token::token, op: ast::binop, prec: int};

// FIXME make this a const, don't store it in parser state
#[doc = "The precedence of binary operators"]
fn binop_prec_table() -> @[op_spec] {
    ret @[{tok: token::BINOP(token::STAR), op: ast::mul, prec: 12},
          {tok: token::BINOP(token::SLASH), op: ast::div, prec: 12},
          {tok: token::BINOP(token::PERCENT), op: ast::rem, prec: 12},
          // 'as' sits between here with 11
          {tok: token::BINOP(token::PLUS), op: ast::add, prec: 10},
          {tok: token::BINOP(token::MINUS), op: ast::subtract, prec: 10},
          {tok: token::BINOP(token::LSL), op: ast::lsl, prec: 9},
          {tok: token::BINOP(token::LSR), op: ast::lsr, prec: 9},
          {tok: token::BINOP(token::ASR), op: ast::asr, prec: 9},
          {tok: token::BINOP(token::AND), op: ast::bitand, prec: 8},
          {tok: token::BINOP(token::CARET), op: ast::bitxor, prec: 7},
          {tok: token::BINOP(token::OR), op: ast::bitor, prec: 6},
          {tok: token::LT, op: ast::lt, prec: 4},
          {tok: token::LE, op: ast::le, prec: 4},
          {tok: token::GE, op: ast::ge, prec: 4},
          {tok: token::GT, op: ast::gt, prec: 4},
          {tok: token::EQEQ, op: ast::eq, prec: 3},
          {tok: token::NE, op: ast::ne, prec: 3},
          {tok: token::ANDAND, op: ast::and, prec: 2},
          {tok: token::OROR, op: ast::or, prec: 1}];
}
