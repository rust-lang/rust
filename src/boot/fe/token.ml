type token =

    (* Expression operator symbols *)
    PLUS
  | MINUS
  | STAR
  | SLASH
  | PERCENT
  | EQ
  | LT
  | LE
  | EQEQ
  | NE
  | GE
  | GT
  | NOT
  | TILDE
  | CARET
  | AND
  | ANDAND
  | OR
  | OROR
  | LSL
  | LSR
  | ASR
  | OPEQ of token
  | AS
  | WITH

  (* Structural symbols *)
  | AT
  | DOT
  | COMMA
  | SEMI
  | COLON
  | RARROW
  | SEND
  | LARROW
  | LPAREN
  | RPAREN
  | LBRACKET
  | RBRACKET
  | LBRACE
  | RBRACE

  (* Module and crate keywords *)
  | MOD
  | USE
  | AUTH
  | META

  (* Metaprogramming keywords *)
  | SYNTAX
  | POUND

  (* Statement keywords *)
  | IF
  | ELSE
  | DO
  | WHILE
  | ALT
  | CASE

  | FAIL
  | DROP

  | IN
  | FOR
  | EACH
  | PUT
  | RET
  | BE

  (* Type and type-state keywords *)
  | TYPE
  | CHECK
  | CLAIM
  | PROVE

  (* Effect keywords *)
  | IO
  | STATE
  | UNSAFE

  (* Type qualifiers *)
  | NATIVE
  | AUTO
  | MUTABLE

  (* Name management *)
  | IMPORT
  | EXPORT

  (* Value / stmt declarators *)
  | LET

  (* Magic runtime services *)
  | LOG
  | SPAWN
  | BIND
  | THREAD
  | YIELD
  | JOIN

  (* Literals *)
  | LIT_INT       of (int64 * string)
  | LIT_FLO       of string
  | LIT_STR       of string
  | LIT_CHAR      of int
  | LIT_BOOL      of bool

  (* Name components *)
  | IDENT         of string
  | IDX           of int
  | UNDERSCORE

  (* Reserved type names *)
  | BOOL
  | INT
  | UINT
  | CHAR
  | STR
  | MACH          of Common.ty_mach

  (* Algebraic type constructors *)
  | REC
  | TUP
  | TAG
  | VEC
  | ANY

  (* Callable type constructors *)
  | FN
  | ITER

  (* Object type *)
  | OBJ

  (* Comm and task types *)
  | CHAN
  | PORT
  | TASK

  | EOF

  | BRACEQUOTE of string

;;

let rec string_of_tok t =
  match t with
      (* Operator symbols (mostly) *)
      PLUS       -> "+"
    | MINUS      -> "-"
    | STAR       -> "*"
    | SLASH      -> "/"
    | PERCENT    -> "%"
    | EQ         -> "="
    | LT         -> "<"
    | LE         -> "<="
    | EQEQ       -> "=="
    | NE         -> "!="
    | GE         -> ">="
    | GT         -> ">"
    | TILDE      -> "~"
    | CARET      -> "^"
    | NOT        -> "!"
    | AND        -> "&"
    | ANDAND     -> "&&"
    | OR         -> "|"
    | OROR       -> "||"
    | LSL        -> "<<"
    | LSR        -> ">>"
    | ASR        -> ">>>"
    | OPEQ op    -> string_of_tok op ^ "="
    | AS         -> "as"
    | WITH       -> "with"

    (* Structural symbols *)
    | AT         -> "@"
    | DOT        -> "."
    | COMMA      -> ","
    | SEMI       -> ";"
    | COLON      -> ":"
    | RARROW     -> "->"
    | SEND       -> "<|"
    | LARROW     -> "<-"
    | LPAREN     -> "("
    | RPAREN     -> ")"
    | LBRACKET   -> "["
    | RBRACKET   -> "]"
    | LBRACE     -> "{"
    | RBRACE     -> "}"

    (* Module and crate keywords *)
    | MOD        -> "mod"
    | USE        -> "use"
    | AUTH       -> "auth"

    (* Metaprogramming keywords *)
    | SYNTAX     -> "syntax"
    | META       -> "meta"
    | POUND      -> "#"

    (* Control-flow keywords *)
    | IF         -> "if"
    | ELSE       -> "else"
    | DO         -> "do"
    | WHILE      -> "while"
    | ALT        -> "alt"
    | CASE       -> "case"

    | FAIL       -> "fail"
    | DROP       -> "drop"

    | IN         -> "in"
    | FOR        -> "for"
    | EACH       -> "each"
    | PUT        -> "put"
    | RET        -> "ret"
    | BE         -> "be"

    (* Type and type-state keywords *)
    | TYPE       -> "type"
    | CHECK      -> "check"
    | CLAIM      -> "claim"
    | PROVE      -> "prove"

    (* Effect keywords *)
    | IO         -> "io"
    | STATE      -> "state"
    | UNSAFE     -> "unsafe"

    (* Type qualifiers *)
    | NATIVE     -> "native"
    | AUTO       -> "auto"
    | MUTABLE    -> "mutable"

    (* Name management *)
    | IMPORT     -> "import"
    | EXPORT     -> "export"

    (* Value / stmt declarators. *)
    | LET        -> "let"

    (* Magic runtime services *)
    | LOG        -> "log"
    | SPAWN      -> "spawn"
    | BIND       -> "bind"
    | THREAD     -> "thread"
    | YIELD      -> "yield"
    | JOIN       -> "join"

    (* Literals *)
    | LIT_INT (_,s)  -> s
    | LIT_FLO n  -> n
    | LIT_STR s  -> ("\"" ^ (String.escaped s) ^ "\"")
    | LIT_CHAR c -> ("'" ^ (Common.escaped_char c) ^ "'")
    | LIT_BOOL b -> if b then "true" else "false"

    (* Name components *)
    | IDENT s    -> s
    | IDX i      -> ("_" ^ (string_of_int i))
    | UNDERSCORE -> "_"

    (* Reserved type names *)
    | BOOL       -> "bool"
    | INT        -> "int"
    | UINT       -> "uint"
    | CHAR       -> "char"
    | STR        -> "str"
    | MACH m     -> Common.string_of_ty_mach m

    (* Algebraic type constructors *)
    | REC        -> "rec"
    | TUP        -> "tup"
    | TAG        -> "tag"
    | VEC        -> "vec"
    | ANY        -> "any"

    (* Callable type constructors *)
    | FN         -> "fn"
    | ITER       -> "fn"

    (* Object type *)
    | OBJ        -> "obj"

    (* Ports and channels *)
    | CHAN          -> "chan"
    | PORT          -> "port"

    (* Taskess types *)
    | TASK         -> "task"

    | BRACEQUOTE _ -> "{...bracequote...}"

    | EOF          -> "<EOF>"
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
