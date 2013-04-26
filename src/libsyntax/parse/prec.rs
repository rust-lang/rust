// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::*;
use parse::token::*;
use parse::token::Token;

/// Unary operators have higher precedence than binary
pub static unop_prec: uint = 100u;

/**
 * Precedence of the `as` operator, which is a binary operator
 * but is not represented in the precedence table.
 */
pub static as_prec: uint = 11u;

/**
 * Maps a token to a record specifying the corresponding binary
 * operator and its precedence
 */
pub fn token_to_binop(tok: Token) -> Option<ast::binop> {
  match tok {
      BINOP(STAR)    => Some(mul),
      BINOP(SLASH)   => Some(quot),
      BINOP(PERCENT) => Some(rem),
      // 'as' sits between here with 11
      BINOP(PLUS)    => Some(add),
      BINOP(MINUS)   => Some(subtract),
      BINOP(SHL)     => Some(shl),
      BINOP(SHR)     => Some(shr),
      BINOP(AND)     => Some(bitand),
      BINOP(CARET)   => Some(bitxor),
      BINOP(OR)      => Some(bitor),
      LT             => Some(lt),
      LE             => Some(le),
      GE             => Some(ge),
      GT             => Some(gt),
      EQEQ           => Some(eq),
      NE             => Some(ne),
      ANDAND         => Some(and),
      OROR           => Some(or),
      _              => None
  }
}
