

{

  open Token;;
  open Common;;

  exception Lex_err of (string * Common.pos);;

  let fail lexbuf s =
    let p = lexbuf.Lexing.lex_start_p in
    let pos =
      (p.Lexing.pos_fname,
       p.Lexing.pos_lnum ,
       (p.Lexing.pos_cnum) - (p.Lexing.pos_bol))
    in
      raise (Lex_err (s, pos))
  ;;

  let bump_line p = { p with
              Lexing.pos_lnum = p.Lexing.pos_lnum + 1;
              Lexing.pos_bol = p.Lexing.pos_cnum }
  ;;

  let keyword_table = Hashtbl.create 100
  let _ =
    List.iter (fun (kwd, tok) -> Common.htab_put keyword_table kwd tok)
              [ ("mod", MOD);
                ("use", USE);
                ("meta", META);
                ("auth", AUTH);

                ("syntax", SYNTAX);

                ("if", IF);
                ("else", ELSE);
                ("while", WHILE);
                ("do", DO);
                ("alt", ALT);
                ("case", CASE);

                ("for", FOR);
                ("each", EACH);
                ("put", PUT);
                ("ret", RET);
                ("be", BE);

                ("fail", FAIL);
                ("drop", DROP);

                ("type", TYPE);
                ("check", CHECK);
                ("claim", CLAIM);
                ("prove", PROVE);

                ("io", IO);
                ("state", STATE);
                ("unsafe", UNSAFE);

                ("native", NATIVE);
                ("mutable", MUTABLE);
                ("auto", AUTO);

                ("fn", FN);
                ("iter", ITER);

                ("import", IMPORT);
                ("export", EXPORT);

                ("let", LET);

                ("log", LOG);
                ("spawn", SPAWN);
                ("thread", THREAD);
                ("yield", YIELD);
                ("join", JOIN);

                ("bool", BOOL);

                ("int", INT);
                ("uint", UINT);

                ("char", CHAR);
                ("str", STR);

                ("rec", REC);
                ("tup", TUP);
                ("tag", TAG);
                ("vec", VEC);
                ("any", ANY);

                ("obj", OBJ);

                ("port", PORT);
                ("chan", CHAN);

                ("task", TASK);

                ("true", LIT_BOOL true);
                ("false", LIT_BOOL false);

                ("in", IN);

                ("as", AS);
                ("with", WITH);

                ("bind", BIND);

                ("u8", MACH TY_u8);
                ("u16", MACH TY_u16);
                ("u32", MACH TY_u32);
                ("u64", MACH TY_u64);
                ("i8", MACH TY_i8);
                ("i16", MACH TY_i16);
                ("i32", MACH TY_i32);
                ("i64", MACH TY_i64);
                ("f32", MACH TY_f32);
                ("f64", MACH TY_f64)
              ]
;;
}

let hexdig = ['0'-'9' 'a'-'f' 'A'-'F']
let bin = "0b" ['0' '1']['0' '1' '_']*
let hex = "0x" hexdig ['0'-'9' 'a'-'f' 'A'-'F' '_']*
let dec = ['0'-'9']+
let exp = ['e''E']['-''+']? dec
let flo = (dec '.' dec (exp?)) | (dec exp)

let ws = [ ' ' '\t' '\r' ]

let id = ['a'-'z' 'A'-'Z' '_']['a'-'z' 'A'-'Z' '0'-'9' '_']*

rule token = parse
  ws+                          { token lexbuf }
| '\n'                         { lexbuf.Lexing.lex_curr_p
                                     <- (bump_line lexbuf.Lexing.lex_curr_p);
                                 token lexbuf }
| "//" [^'\n']*                { token lexbuf }

| '+'                          { PLUS       }
| '-'                          { MINUS      }
| '*'                          { STAR       }
| '/'                          { SLASH      }
| '%'                          { PERCENT    }
| '='                          { EQ         }
| '<'                          { LT         }
| "<="                         { LE         }
| "=="                         { EQEQ       }
| "!="                         { NE         }
| ">="                         { GE         }
| '>'                          { GT         }
| '!'                          { NOT        }
| '&'                          { AND        }
| "&&"                         { ANDAND     }
| '|'                          { OR         }
| "||"                         { OROR       }
| "<<"                         { LSL        }
| ">>"                         { LSR        }
| ">>>"                        { ASR        }
| '~'                          { TILDE      }
| '{'                          { LBRACE     }
| '_' (dec as n)               { IDX (int_of_string n) }
| '_'                          { UNDERSCORE }
| '}'                          { RBRACE     }

| "+="                         { OPEQ (PLUS)    }
| "-="                         { OPEQ (MINUS)   }
| "*="                         { OPEQ (STAR)    }
| "/="                         { OPEQ (SLASH)   }
| "%="                         { OPEQ (PERCENT) }
| "&="                         { OPEQ (AND) }
| "|="                         { OPEQ (OR)  }
| "<<="                        { OPEQ (LSL) }
| ">>="                        { OPEQ (LSR) }
| ">>>="                       { OPEQ (ASR) }
| "^="                         { OPEQ (CARET) }

| '#'                          { POUND      }
| '@'                          { AT         }
| '^'                          { CARET      }
| '.'                          { DOT        }
| ','                          { COMMA      }
| ';'                          { SEMI       }
| ':'                          { COLON      }
| "<-"                         { LARROW     }
| "<|"                         { SEND       }
| "->"                         { RARROW     }
| '('                          { LPAREN     }
| ')'                          { RPAREN     }
| '['                          { LBRACKET   }
| ']'                          { RBRACKET   }

| id as i
                               { try
                                     Hashtbl.find keyword_table i
                                 with
                                     Not_found -> IDENT (i)
                                            }

| bin as n                      { LIT_INT (Int64.of_string n, n)    }
| hex as n                      { LIT_INT (Int64.of_string n, n)    }
| dec as n                      { LIT_INT (Int64.of_string n, n)    }
| flo as n                      { LIT_FLO n                         }

| '\''                          { char lexbuf                       }
| '"'                           { let buf = Buffer.create 32 in
                                    str buf lexbuf                  }

| eof                           { EOF        }

and str buf = parse
    _ as ch
    {
      match ch with
          '"' -> LIT_STR (Buffer.contents buf)
        | '\\' -> str_escape buf lexbuf
        | _ ->
            Buffer.add_char buf ch;
            let c = Char.code ch in
              if bounds 0 c 0x7f
              then str buf lexbuf
              else
                if ((c land 0b1110_0000) == 0b1100_0000)
                then ext_str 1 buf lexbuf
                else
                  if ((c land 0b1111_0000) == 0b1110_0000)
                  then ext_str 2 buf lexbuf
                  else
                    if ((c land 0b1111_1000) == 0b1111_0000)
                    then ext_str 3 buf lexbuf
                    else
                      if ((c land 0b1111_1100) == 0b1111_1000)
                      then ext_str 4 buf lexbuf
                      else
                        if ((c land 0b1111_1110) == 0b1111_1100)
                        then ext_str 5 buf lexbuf
                        else fail lexbuf "bad initial utf-8 byte"
    }

and str_escape buf = parse
    'x' ((hexdig hexdig) as h)
  | 'u' ((hexdig hexdig hexdig hexdig) as h)
  | 'U'
      ((hexdig hexdig hexdig hexdig
        hexdig hexdig hexdig hexdig) as h)
      {
        Buffer.add_string buf (char_as_utf8 (int_of_string ("0x" ^ h)));
        str buf lexbuf
      }
  | 'n' { Buffer.add_char buf '\n'; str buf lexbuf }
  | 'r' { Buffer.add_char buf '\r'; str buf lexbuf }
  | 't' { Buffer.add_char buf '\t'; str buf lexbuf }
  | '\\' { Buffer.add_char buf '\\'; str buf lexbuf }
  | '"' { Buffer.add_char buf '"'; str buf lexbuf }
  | _ as c { fail lexbuf ("bad escape: \\" ^ (Char.escaped c))  }


and ext_str n buf = parse
    _ as ch
      {
        let c = Char.code ch in
          if ((c land 0b1100_0000) == (0b1000_0000))
          then
            begin
              Buffer.add_char buf ch;
              if n = 1
              then str buf lexbuf
              else ext_str (n-1) buf lexbuf
            end
          else
            fail lexbuf "bad trailing utf-8 byte"
      }


and char = parse
    '\\' { char_escape lexbuf }
  | _ as c
    {
      let c = Char.code c in
        if bounds 0 c 0x7f
        then end_char c lexbuf
        else
          if ((c land 0b1110_0000) == 0b1100_0000)
          then ext_char 1 (c land 0b0001_1111) lexbuf
          else
            if ((c land 0b1111_0000) == 0b1110_0000)
            then ext_char 2 (c land 0b0000_1111) lexbuf
            else
              if ((c land 0b1111_1000) == 0b1111_0000)
              then ext_char 3 (c land 0b0000_0111) lexbuf
              else
                if ((c land 0b1111_1100) == 0b1111_1000)
                then ext_char 4 (c land 0b0000_0011) lexbuf
                else
                  if ((c land 0b1111_1110) == 0b1111_1100)
                  then ext_char 5 (c land 0b0000_0001) lexbuf
                  else fail lexbuf "bad initial utf-8 byte"
    }

and char_escape = parse
    'x' ((hexdig hexdig) as h)
  | 'u' ((hexdig hexdig hexdig hexdig) as h)
  | 'U'
      ((hexdig hexdig hexdig hexdig
        hexdig hexdig hexdig hexdig) as h)
      {
        end_char (int_of_string ("0x" ^ h)) lexbuf
      }
  | 'n' { end_char (Char.code '\n') lexbuf }
  | 'r' { end_char (Char.code '\r') lexbuf }
  | 't' { end_char (Char.code '\t') lexbuf }
  | '\\' { end_char (Char.code '\\') lexbuf }
  | '\'' { end_char (Char.code '\'') lexbuf }
  | _ as c { fail lexbuf ("bad escape: \\" ^ (Char.escaped c))  }


and ext_char n accum = parse
  _ as c
    {
      let c = Char.code c in
        if ((c land 0b1100_0000) == (0b1000_0000))
        then
          let accum = (accum lsl 6) lor (c land 0b0011_1111) in
            if n = 1
            then end_char accum lexbuf
            else ext_char (n-1) accum lexbuf
        else
          fail lexbuf "bad trailing utf-8 byte"
    }

and end_char accum = parse
  '\'' { LIT_CHAR accum }


and bracequote buf depth = parse

  '\\' '{'                      { Buffer.add_char buf '{';
                                  bracequote buf depth lexbuf          }

| '{'                           { Buffer.add_char buf '{';
                                  bracequote buf (depth+1) lexbuf      }

| '\\' '}'                      { Buffer.add_char buf '}';
                                  bracequote buf depth lexbuf          }

| '}'                           { if depth = 1
                                  then BRACEQUOTE (Buffer.contents buf)
                                  else
                                    begin
                                      Buffer.add_char buf '}';
                                      bracequote buf (depth-1) lexbuf
                                    end                                }

| '\\' [^'{' '}']               { let s = Lexing.lexeme lexbuf in
                                    Buffer.add_string buf s;
                                    bracequote buf depth lexbuf        }


| [^'\\' '{' '}']+              { let s = Lexing.lexeme lexbuf in
                                    Buffer.add_string buf s;
                                    bracequote buf depth lexbuf        }
