(* The 'fmt' extension is modeled on the posix printf system.
 * 
 * A posix conversion ostensibly looks like this:
 * 
 * %[parameter][flags][width][.precision][length]type
 * 
 * Given the different numeric type bestiary we have, we omit the 'length'
 * parameter and support slightly different conversions for 'type':
 * 
 * %[parameter][flags][width][.precision]type
 * 
 * we also only support translating-to-rust a tiny subset of the possible
 * combinations at the moment.
 *)

exception Malformed of string
;;

type case =
    CASE_upper
  | CASE_lower
;;

type signedness =
    SIGNED
  | UNSIGNED
;;

type ty =
    TY_bool
  | TY_str
  | TY_char
  | TY_int of signedness
  | TY_bits
  | TY_hex of case
      (* FIXME: Support more later. *)
;;

type flag =
    FLAG_left_justify
  | FLAG_left_zero_pad
  | FLAG_left_space_pad
  | FLAG_plus_if_positive
  | FLAG_alternate
;;

type count =
    COUNT_is of int
  | COUNT_is_param of int
  | COUNT_is_next_param
  | COUNT_implied

type conv =
    { conv_parameter: int option;
      conv_flags: flag list;
      conv_width: count;
      conv_precision: count;
      conv_ty: ty }

type piece =
    PIECE_string of string
  | PIECE_conversion of conv


let rec peek_num (s:string) (i:int) (lim:int)
    : (int * int) option =
  if i >= lim
  then None
  else
    let c = s.[i] in
      if '0' <= c && c <= '9'
      then
        let n = (Char.code c) - (Char.code '0') in
          match peek_num s (i+1) lim with
              None -> Some (n, i+1)
            | Some (m, i) -> Some (n * 10 + m, i)
      else None
;;

let parse_parameter (s:string) (i:int) (lim:int)
    : (int option * int) =
  if i >= lim
  then (None, i)
  else
    match peek_num s i lim with
        None -> (None, i)
      | Some (n, j) ->
          if j < (String.length s) && s.[j] = '$'
          then (Some n, j+1)
          else (None, i)
;;

let rec parse_flags (s:string) (i:int) (lim:int)
    : (flag list * int) =
  if i >= lim
  then ([], i)
  else
    let cont flag =
      let (rest, j) = parse_flags s (i+1) lim in
        (flag :: rest, j)
    in
      match s.[i] with
          '-' -> cont FLAG_left_justify
        | '0' -> cont FLAG_left_zero_pad
        | ' ' -> cont FLAG_left_space_pad
        | '+' -> cont FLAG_plus_if_positive
        | '#' -> cont FLAG_alternate
        | _ -> ([], i)
;;

let parse_count (s:string) (i:int) (lim:int)
    : (count * int) =
  if i >= lim
  then (COUNT_implied, i)
  else
    if s.[i] = '*'
    then
      begin
        match parse_parameter s (i+1) lim with
            (None, j) -> (COUNT_is_next_param, j)
          | (Some n, j) -> (COUNT_is_param n, j)
      end
    else
      begin
        match peek_num s i lim with
            None -> (COUNT_implied, i)
          | Some (n, j) -> (COUNT_is n, j)
      end
;;

let parse_precision (s:string) (i:int) (lim:int)
    : (count * int) =
  if i >= lim
  then (COUNT_implied, i)
  else
    if s.[i] = '.'
    then parse_count s (i+1) lim
    else (COUNT_implied, i)
;;

let parse_type (s:string) (i:int) (lim:int)
    : (ty * int) =
  if i >= lim
  then raise (Malformed "missing type in conversion")
  else
    let t =
      match s.[i] with
          'b' -> TY_bool
        | 's' -> TY_str
        | 'c' -> TY_char
        | 'd' | 'i' -> TY_int SIGNED
        | 'u' -> TY_int UNSIGNED
        | 'x' -> TY_hex CASE_lower
        | 'X' -> TY_hex CASE_upper
        | 't' -> TY_bits
        | _ -> raise (Malformed "unknown type in conversion")
    in
      (t, i+1)
;;

let parse_conversion (s:string) (i:int) (lim:int)
    : (piece * int) =
  let (parameter, i) = parse_parameter s i lim in
  let (flags, i) = parse_flags s i lim in
  let (width, i) = parse_count s i lim in
  let (precision, i) = parse_precision s i lim in
  let (ty, i) = parse_type s i lim in
    (PIECE_conversion  { conv_parameter = parameter;
                         conv_flags = flags;
                         conv_width = width;
                         conv_precision = precision;
                         conv_ty = ty }, i)
;;

let parse_fmt_string (s:string) : piece array =
  let pieces = Queue.create () in
  let i = ref 0 in
  let lim = String.length s in
  let buf = Buffer.create 10 in
  let flush_buf _ =
    if (Buffer.length buf) <> 0
    then
      let piece =
        PIECE_string (Buffer.contents buf)
      in
        Queue.add piece pieces;
        Buffer.clear buf;
  in
    while (!i) < lim
    do
      if s.[!i] = '%'
      then
        begin
          incr i;
          if (!i) >= lim
          then raise (Malformed "unterminated conversion at end of string");
          if s.[!i] = '%'
          then
            begin
              Buffer.add_char buf '%';
              incr i;
            end
          else
            begin
              flush_buf();
              let (piece, j) = parse_conversion s (!i) lim in
                Queue.add piece pieces;
                i := j
            end
        end
      else
        begin
          Buffer.add_char buf s.[!i];
          incr i;
        end
    done;
    flush_buf ();
    Common.queue_to_arr pieces
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
