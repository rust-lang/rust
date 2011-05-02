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
  | QUES
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
  | BREAK
  | CONT

  (* Type and type-state keywords *)
  | TYPE
  | CHECK
  | ASSERT
  | CLAIM
  | PROVE

  (* Layer keywords *)
  | STATE
  | GC

  (* Unsafe-block keyword *)
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
  | CONST

  (* Magic runtime services *)
  | LOG
  | LOG_ERR
  | SPAWN
  | BIND
  | THREAD
  | YIELD
  | JOIN

  (* Literals *)
  | LIT_INT       of int64
  | LIT_UINT      of int64
  | LIT_FLOAT     of float
  | LIT_MACH_INT  of Common.ty_mach * int64
  | LIT_MACH_FLOAT of Common.ty_mach * float
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
  | FLOAT
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
    | QUES       -> "?"
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
    | BREAK      -> "break"
    | CONT       -> "cont"

    (* Type and type-state keywords *)
    | TYPE       -> "type"
    | CHECK      -> "check"
    | ASSERT     -> "assert"
    | CLAIM      -> "claim"
    | PROVE      -> "prove"

    (* Layer keywords *)
    | STATE      -> "state"
    | GC         -> "gc"

    (* Unsafe-block keyword *)
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
    | CONST      -> "const"

    (* Magic runtime services *)
    | LOG        -> "log"
    | LOG_ERR    -> "log_err"
    | SPAWN      -> "spawn"
    | BIND       -> "bind"
    | THREAD     -> "thread"
    | YIELD      -> "yield"
    | JOIN       -> "join"

    (* Literals *)
    | LIT_INT i  -> Int64.to_string i
    | LIT_UINT i -> (Int64.to_string i) ^ "u"
    | LIT_FLOAT s  -> string_of_float s
    | LIT_MACH_INT (tm, i)  ->
        (Int64.to_string i) ^ (Common.string_of_ty_mach tm)
    | LIT_MACH_FLOAT (tm, f)  ->
        (string_of_float f) ^ (Common.string_of_ty_mach tm)
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
    | FLOAT      -> "float"
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
    | ITER       -> "iter"

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
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
