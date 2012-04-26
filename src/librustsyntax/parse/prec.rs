export as_prec;
export unop_prec;
export token_to_binop;

import token::*;
import token::token;
import ast::*;

#[doc = "Unary operators have higher precedence than binary"]
const unop_prec: uint = 100u;

#[doc = "
Precedence of the `as` operator, which is a binary operator
but is not represented in the precedence table.
"]
const as_prec: uint = 11u;

#[doc = "Maps a token to a record specifying the corresponding binary
         operator and its precedence"]
fn token_to_binop(tok: token) -> option<ast::binop> {
  alt tok {
      BINOP(STAR)    { some(mul) }
      BINOP(SLASH)   { some(div) }
      BINOP(PERCENT) { some(rem) }
      // 'as' sits between here with 11
      BINOP(PLUS)    { some(add) }
      BINOP(MINUS)   { some(subtract) }
      BINOP(LSL)     { some(lsl) }
      BINOP(LSR)     { some(lsr) }
      BINOP(ASR)     { some(asr) }
      BINOP(AND)     { some(bitand) }
      BINOP(CARET)   { some(bitxor) }
      BINOP(OR)      { some(bitor) }
      LT             { some(lt) }
      LE             { some(le) }
      GE             { some(ge) }
      GT             { some(gt) }
      EQEQ           { some(eq) }
      NE             { some(ne) }
      ANDAND         { some(and) }
      OROR           { some(or) }
      _              { none }
  }
}
