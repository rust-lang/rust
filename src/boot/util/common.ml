(*
 * This module goes near the *bottom* of the dependency DAG, and holds basic
 * types shared across all phases of the compiler.
 *)

type ('a, 'b) either = Left of 'a | Right of 'b

type filename = string
type pos = (filename * int * int)
type span = {lo: pos; hi: pos}

type node_id = Node of int
type temp_id = Temp of int
type opaque_id = Opaque of int
type constr_id = Constr of int
type crate_id = Crate of int

let int_of_node (Node i) = i
let int_of_temp (Temp i) = i
let int_of_opaque (Opaque i) = i
let int_of_constr (Constr i) = i
let int_of_common (Crate i) = i

type 'a identified = { node: 'a; id: node_id }
;;

let bug _ =
  let k s = failwith s
  in Printf.ksprintf k
;;

(* TODO: On some joyous day, remove me. *)
exception Not_implemented of ((node_id option) * string)
;;

exception Semant_err of ((node_id option) * string)
;;

let err (idopt:node_id option) =
  let k s =
    raise (Semant_err (idopt, s))
  in
    Printf.ksprintf k
;;

let unimpl (idopt:node_id option) =
  let k s =
    raise (Not_implemented (idopt, "unimplemented " ^ s))
  in
    Printf.ksprintf k
;;

(* Some ubiquitous low-level types. *)

type target =
    Linux_x86_elf
  | Win32_x86_pe
  | MacOS_x86_macho
  | FreeBSD_x86_elf
;;

type ty_mach =
    TY_u8
  | TY_u16
  | TY_u32
  | TY_u64
  | TY_i8
  | TY_i16
  | TY_i32
  | TY_i64
  | TY_f32
  | TY_f64
;;

let mach_is_integral (mach:ty_mach) : bool =
  match mach with
      TY_i8 | TY_i16 | TY_i32 | TY_i64
    | TY_u8 | TY_u16 | TY_u32 | TY_u64 -> true
    | TY_f32 | TY_f64 -> false
;;


let mach_is_signed (mach:ty_mach) : bool =
  match mach with
      TY_i8 | TY_i16 | TY_i32 | TY_i64 -> true
    | TY_u8 | TY_u16 | TY_u32 | TY_u64
    | TY_f32 | TY_f64 -> false
;;

let string_of_ty_mach (mach:ty_mach) : string =
  match mach with
    TY_u8 -> "u8"
  | TY_u16 -> "u16"
  | TY_u32 -> "u32"
  | TY_u64 -> "u64"
  | TY_i8 -> "i8"
  | TY_i16 -> "i16"
  | TY_i32 -> "i32"
  | TY_i64 -> "i64"
  | TY_f32 -> "f32"
  | TY_f64 -> "f64"
;;

let bytes_of_ty_mach (mach:ty_mach) : int =
  match mach with
    TY_u8 -> 1
  | TY_u16 -> 2
  | TY_u32 -> 4
  | TY_u64 -> 8
  | TY_i8 -> 1
  | TY_i16 -> 2
  | TY_i32 -> 4
  | TY_i64 -> 8
  | TY_f32 -> 4
  | TY_f64 -> 8
;;

type ty_param_idx = int
;;

type nabi_conv =
    CONV_rust
  | CONV_cdecl
;;

type nabi = { nabi_indirect: bool;
              nabi_convention: nabi_conv }
;;

let string_to_conv (a:string) : nabi_conv option =
  match a with
      "cdecl" -> Some CONV_cdecl
    | "rust" -> Some CONV_rust
    | _ -> None

(* FIXME: remove this when native items go away. *)
let string_to_nabi (s:string) (indirect:bool) : nabi option =
  match string_to_conv s with
      None -> None
    | Some c ->
        Some { nabi_indirect = indirect;
               nabi_convention = c }
;;

type required_lib_spec =
    {
      required_libname: string;
      required_prefix: int;
    }
;;

type required_lib =
    REQUIRED_LIB_rustrt
  | REQUIRED_LIB_crt
  | REQUIRED_LIB_rust of required_lib_spec
  | REQUIRED_LIB_c of required_lib_spec
;;

type segment =
    SEG_text
  | SEG_data
;;

type fixup =
    { fixup_name: string;
      mutable fixup_file_pos: int option;
      mutable fixup_file_sz: int option;
      mutable fixup_mem_pos: int64 option;
      mutable fixup_mem_sz: int64 option }
;;


let new_fixup (s:string)
    : fixup =
  { fixup_name = s;
    fixup_file_pos = None;
    fixup_file_sz = None;
    fixup_mem_pos = None;
    fixup_mem_sz = None }
;;


(*
 * Auxiliary string functions.
 *)

let split_string (c:char) (s:string) : string list =
  let ls = ref [] in
  let b = Buffer.create (String.length s) in
  let flush _ =
    if Buffer.length b <> 0
    then
      begin
        ls := (Buffer.contents b) :: (!ls);
        Buffer.clear b
      end
  in
  let f ch =
    if c = ch
    then flush()
    else Buffer.add_char b ch
  in
    String.iter f s;
    flush();
    List.rev (!ls)
;;

(*
 * Auxiliary hashtable functions.
 *)

let htab_keys (htab:('a,'b) Hashtbl.t) : ('a list) =
  Hashtbl.fold (fun k _ accum -> k :: accum) htab []
;;

let sorted_htab_keys (tab:('a, 'b) Hashtbl.t) : 'a array =
  let keys = Array.of_list (htab_keys tab) in
    Array.sort compare keys;
    keys
;;

let sorted_htab_iter
    (f:'a -> 'b -> unit)
    (tab:('a, 'b) Hashtbl.t)
    : unit =
  Array.iter
    (fun k -> f k (Hashtbl.find tab k))
    (sorted_htab_keys tab)
;;

let htab_vals (htab:('a,'b) Hashtbl.t) : ('b list)  =
  Hashtbl.fold (fun _ v accum -> v :: accum) htab []
;;

let htab_pairs (htab:('a,'b) Hashtbl.t) : (('a * 'b) list) =
  Hashtbl.fold (fun k v accum -> (k,v) :: accum) htab []
;;

let htab_search (htab:('a,'b) Hashtbl.t) (k:'a) : ('b option) =
  if Hashtbl.mem htab k
  then Some (Hashtbl.find htab k)
  else None
;;

let htab_search_or_default
    (htab:('a,'b) Hashtbl.t)
    (k:'a)
    (def:unit -> 'b)
    : 'b =
  match htab_search htab k with
      Some v -> v
    | None -> def()
;;

let htab_search_or_add
    (htab:('a,'b) Hashtbl.t)
    (k:'a)
    (mk:unit -> 'b)
    : 'b =
  let def () =
    let v = mk() in
      Hashtbl.add htab k v;
      v
  in
    htab_search_or_default htab k def
;;

let htab_put (htab:('a,'b) Hashtbl.t) (a:'a) (b:'b) : unit =
  assert (not (Hashtbl.mem htab a));
  Hashtbl.add htab a b
;;

(* This is completely ridiculous, but it turns out that ocaml hashtables are
 * order-of-element-addition sensitive when it comes to the built-in
 * polymorphic comparison operator. So you have to canonicalize them after
 * you've stopped adding things to them if you ever want to use them in a
 * term that requires structural comparison to work. Sigh.
 *)

let htab_canonicalize (htab:('a,'b) Hashtbl.t) : ('a,'b) Hashtbl.t =
  let n = Hashtbl.create (Hashtbl.length htab) in
    Array.iter
      (fun k -> Hashtbl.add n k (Hashtbl.find htab k))
      (sorted_htab_keys htab);
    n
;;

let htab_map
    (htab:('a,'b) Hashtbl.t)
    (f:'a -> 'b -> ('c * 'd))
    : (('c,'d) Hashtbl.t) =
  let ntab = Hashtbl.create (Hashtbl.length htab) in
  let g a b =
    let (c,d) = f a b in
      htab_put ntab c d
  in
    Hashtbl.iter g htab;
    htab_canonicalize (ntab)
;;

let htab_fold
    (fn:'a -> 'b -> 'c -> 'c)
    (init:'c)
    (h:('a, 'b) Hashtbl.t) : 'c =
  let accum = ref init in
  let f a b = accum := (fn a b (!accum)) in
    Hashtbl.iter f h;
    !accum
;;


let reduce_hash_to_list
    (fn:'a -> 'b -> 'c)
    (h:('a, 'b) Hashtbl.t)
    : ('c list) =
  htab_fold (fun a b ls -> (fn a b) :: ls) [] h
;;

(* 
 * Auxiliary association-array and association-list operations.
 *)
let atab_search (atab:('a * 'b) array) (a:'a) : ('b option) =
  let lim = Array.length atab in
  let rec step i =
    if i = lim
    then None
    else
      let (k,v) = atab.(i) in
        if k = a
        then Some v
        else step (i+1)
  in
    step 0

let atab_find (atab:('a * 'b) array) (a:'a) : 'b =
  match atab_search atab a with
      None -> bug () "atab_find: element not found"
    | Some b -> b

let atab_mem (atab:('a * 'b) array) (a:'a) : bool =
  match atab_search atab a with
      None -> false
    | Some _ -> true

let rec ltab_search (ltab:('a * 'b) list) (a:'a) : ('b option) =
  match ltab with
      [] -> None
    | (k,v)::_ when k = a -> Some v
    | _::lz -> ltab_search lz a

let ltab_put (ltab:('a * 'b) list) (a:'a) (b:'b) : (('a * 'b) list) =
  assert ((ltab_search ltab a) = None);
  (a,b)::ltab

(*
 * Auxiliary list functions.
 *)

let rec list_search (list:'a list) (f:'a -> 'b option) : ('b option) =
  match list with
      [] -> None
    | a::az ->
        match f a with
            Some b -> Some b
          | None -> list_search az f

let rec list_search_ctxt
    (list:'a list)
    (f:'a -> 'b option)
    : ((('a list) * 'b) option) =
  match list with
      [] -> None
    | a::az ->
        match f a with
            Some b -> Some (list, b)
          | None -> list_search_ctxt az f

let rec list_drop n ls =
  if n = 0
  then ls
  else list_drop (n-1) (List.tl ls)
;;

let rec list_count elem lst =
  match lst with
      [] -> 0
    | h::t when h = elem -> 1 + (list_count elem t)
    | _::t -> list_count elem t
;;


(*
 * Auxiliary pair functions.
 *)
let pair_rev (x,y) = (y,x)

(*
 * Auxiliary option functions.
 *)

let bool_of_option x =
  match x with
      Some _ -> true
    | None -> false

let may f x =
  match x with
      Some x' -> f x'
    | None -> ()

let option_map f x =
  match x with
      Some x' -> Some (f x')
    | None -> None

let option_get x =
  match x with
      Some x -> x
    | None -> raise Not_found

(*
 * Auxiliary either functions.
 *)
let either_has_left x =
  match x with
      Left _ -> true
    | Right _ -> false
        
let either_has_right x = not (either_has_left x)

let either_get_left x =
  match x with
      Left x -> x
    | Right _ -> raise Not_found

let either_get_right x =
  match x with
      Right x -> x
    | Left _ -> raise Not_found
(*
 * Auxiliary stack functions.
 *)

let stk_fold (s:'a Stack.t) (f:'a -> 'b -> 'b) (x:'b) : 'b =
  let r = ref x in
    Stack.iter (fun e -> r := f e (!r)) s;
    !r

let stk_elts_from_bot (s:'a Stack.t) : ('a list) =
  stk_fold s (fun x y -> x::y) []

let stk_elts_from_top (s:'a Stack.t) : ('a list) =
  List.rev (stk_elts_from_bot s)

let stk_search (s:'a Stack.t) (f:'a -> 'b option) : 'b option =
  stk_fold s (fun e accum -> match accum with None -> (f e) | x -> x) None


(*
 * Auxiliary array functions.
 *)

let arr_search (a:'a array) (f:int -> 'a -> 'b option) : 'b option =
  let max = Array.length a in
  let rec iter i =
    if i < max
    then
      let v = a.(i) in
      let r = f i v in
        match r with
            Some _ -> r
          | None -> iter (i+1)
    else
      None
  in
    iter 0
;;

let arr_idx (arr:'a array) (a:'a) : int =
  let find i v = if v = a then Some i else None in
    match arr_search arr find with
        None -> bug () "arr_idx: element not found"
      | Some i -> i
;;

let arr_map_partial (a:'a array) (f:'a -> 'b option) : 'b array =
  let accum a ls =
    match f a with
        None -> ls
      | Some b -> b :: ls
  in
    Array.of_list (Array.fold_right accum a [])
;;

let arr_filter_some (a:'a option array) : 'a array =
  arr_map_partial a (fun x -> x)
;;

let arr_find_dups (a:'a array) : ('a * 'a) option =
  let copy = Array.copy a in
    Array.sort compare copy;
    let lasti = (Array.length copy) - 1 in
    let rec find_dups i =
      if i < lasti then
        let this = copy.(i) in
        let next = copy.(i+1) in
          (if (this = next) then
             Some (this, next)
           else
             find_dups (i+1))
      else
        None
    in
      find_dups 0
;;

let arr_check_dups (a:'a array) (f:'a -> 'a -> unit) : unit =
  match arr_find_dups a with
      Some (x, y) -> f x y
    | None -> ()
;;

let arr_map2 (f:'a -> 'b -> 'c) (a:'a array) (b:'b array) : 'c array =
  assert ((Array.length a) = (Array.length b));
  Array.init (Array.length a) (fun i -> f a.(i) b.(i))
;;

let arr_iter2 (f:'a -> 'b -> unit) (a:'a array) (b:'b array) : unit =
  assert ((Array.length a) = (Array.length b));
  Array.iteri (fun i a_elem -> f a_elem b.(i)) a
;;

let arr_for_all (f:int -> 'a -> bool) (a:'a array) : bool =
  let len = Array.length a in
  let rec loop i =
    (i >= len) || ((f i a.(i)) && (loop (i+1)))
  in
    loop 0
;;

let arr_exists (f:int -> 'a -> bool) (a:'a array) : bool =
  let len = Array.length a in
  let rec loop i =
    (i < len) && ((f i a.(i)) || (loop (i+1)))
  in
    loop 0
;;

(* 
 * Auxiliary queue functions. 
 *)

let queue_to_list (q:'a Queue.t) : 'a list =
  List.rev (Queue.fold (fun ls elt -> elt :: ls)  []  q)
;;

let queue_to_arr (q:'a Queue.t) : 'a array =
  Array.init (Queue.length q) (fun _ -> Queue.take q)
;;

(*
 * Auxiliary int64 functions
 *)

let i64_lt (a:int64) (b:int64) : bool = (Int64.compare a b) < 0
let i64_le (a:int64) (b:int64) : bool = (Int64.compare a b) <= 0
let i64_ge (a:int64) (b:int64) : bool = (Int64.compare a b) >= 0
let i64_gt (a:int64) (b:int64) : bool = (Int64.compare a b) > 0
let i64_max (a:int64) (b:int64) : int64 =
  (if (Int64.compare a b) > 0 then a else b)
let i64_min (a:int64) (b:int64) : int64 =
  (if (Int64.compare a b) < 0 then a else b)
let i64_align (align:int64) (v:int64) : int64 =
  (assert (align <> 0L));
  let mask = Int64.sub align 1L in
    Int64.logand (Int64.lognot mask) (Int64.add v mask)
;;

let rec i64_for (lo:int64) (hi:int64) (thunk:int64 -> unit) : unit =
  if i64_lt lo hi then
    begin
      thunk lo;
      i64_for (Int64.add lo 1L) hi thunk;
    end
;;

let rec i64_for_rev (hi:int64) (lo:int64) (thunk:int64 -> unit) : unit =
  if i64_ge hi lo then
    begin
      thunk hi;
      i64_for_rev (Int64.sub hi 1L) lo thunk;
    end
;;


(*
 * Auxiliary int32 functions
 *)

let i32_lt (a:int32) (b:int32) : bool = (Int32.compare a b) < 0
let i32_le (a:int32) (b:int32) : bool = (Int32.compare a b) <= 0
let i32_ge (a:int32) (b:int32) : bool = (Int32.compare a b) >= 0
let i32_gt (a:int32) (b:int32) : bool = (Int32.compare a b) > 0
let i32_max (a:int32) (b:int32) : int32 =
  (if (Int32.compare a b) > 0 then a else b)
let i32_min (a:int32) (b:int32) : int32 =
  (if (Int32.compare a b) < 0 then a else b)
let i32_align (align:int32) (v:int32) : int32 =
  (assert (align <> 0l));
  let mask = Int32.sub align 1l in
    Int32.logand (Int32.lognot mask) (Int32.add v mask)
;;

(*
 * Int-as-unichar functions.
 *)

let bounds lo c hi = (lo <= c) && (c <= hi)
;;

let escaped_char i =
  if bounds 0 i 0x7f
  then Char.escaped (Char.chr i)
  else
    if bounds 0 i 0xffff
    then Printf.sprintf "\\u%4.4X" i
    else Printf.sprintf "\\U%8.8X" i
;;

let char_as_utf8 i =
  let buf = Buffer.create 8 in
  let addb i =
    Buffer.add_char buf (Char.chr (i land 0xff))
  in
  let fini _ =
    Buffer.contents buf
  in
  let rec add_trailing_bytes n i =
    if n = 0
    then fini()
    else
      begin
        addb (0b1000_0000 lor ((i lsr ((n-1) * 6)) land 0b11_1111));
        add_trailing_bytes (n-1) i
      end
  in
    if bounds 0 i 0x7f
    then (addb i; fini())
    else
      if bounds 0x80 i 0x7ff
      then (addb ((0b1100_0000) lor (i lsr 6));
            add_trailing_bytes 1 i)
      else
        if bounds 0x800 i 0xffff
        then (addb ((0b1110_0000) lor (i lsr 12));
              add_trailing_bytes 2 i)
        else
          if bounds 0x1000 i 0x1f_ffff
          then (addb ((0b1111_0000) lor (i lsr 18));
                add_trailing_bytes 3 i)
          else
            if bounds 0x20_0000 i 0x3ff_ffff
            then (addb ((0b1111_1000) lor (i lsr 24));
                  add_trailing_bytes 4 i)
            else
              if bounds 0x400_0000 i 0x7fff_ffff
              then (addb ((0b1111_1100) lor (i lsr 30));
                    add_trailing_bytes 5 i)
              else bug () "bad unicode character 0x%X" i
;;

(*
 * Size-expressions.
 *)


type size =
    SIZE_fixed of int64
  | SIZE_fixup_mem_sz of fixup
  | SIZE_fixup_mem_pos of fixup
  | SIZE_param_size of ty_param_idx
  | SIZE_param_align of ty_param_idx
  | SIZE_rt_neg of size
  | SIZE_rt_add of size * size
  | SIZE_rt_mul of size * size
  | SIZE_rt_max of size * size
  | SIZE_rt_align of size * size
;;

let rec string_of_size (s:size) : string =
  match s with
      SIZE_fixed i -> Printf.sprintf "%Ld" i
    | SIZE_fixup_mem_sz f -> Printf.sprintf "%s.mem_sz" f.fixup_name
    | SIZE_fixup_mem_pos f -> Printf.sprintf "%s.mem_pos" f.fixup_name
    | SIZE_param_size i -> Printf.sprintf "ty[%d].size" i
    | SIZE_param_align i -> Printf.sprintf "ty[%d].align" i
    | SIZE_rt_neg a ->
        Printf.sprintf "-(%s)" (string_of_size a)
    | SIZE_rt_add (a, b) ->
        Printf.sprintf "(%s + %s)" (string_of_size a) (string_of_size b)
    | SIZE_rt_mul (a, b) ->
        Printf.sprintf "(%s * %s)" (string_of_size a) (string_of_size b)
    | SIZE_rt_max (a, b) ->
        Printf.sprintf "max(%s,%s)" (string_of_size a) (string_of_size b)
    | SIZE_rt_align (align, off) ->
        Printf.sprintf "align(%s,%s)"
          (string_of_size align) (string_of_size off)
;;

let neg_sz (a:size) : size =
  match a with
      SIZE_fixed a -> SIZE_fixed (Int64.neg a)
    | _ -> SIZE_rt_neg a
;;

let add_sz (a:size) (b:size) : size =
  match (a, b) with
      (SIZE_fixed a, SIZE_fixed b) -> SIZE_fixed (Int64.add a b)

    | ((SIZE_rt_add ((SIZE_fixed a), c)), SIZE_fixed b)
    | ((SIZE_rt_add (c, (SIZE_fixed a))), SIZE_fixed b)
    | (SIZE_fixed a, (SIZE_rt_add ((SIZE_fixed b), c)))
    | (SIZE_fixed a, (SIZE_rt_add (c, (SIZE_fixed b)))) ->
        SIZE_rt_add (SIZE_fixed (Int64.add a b), c)

    | (SIZE_fixed 0L, b) -> b
    | (a, SIZE_fixed 0L) -> a
    | (a, SIZE_fixed b) -> SIZE_rt_add (SIZE_fixed b, a)
    | (a, b) -> SIZE_rt_add (a, b)
;;

let mul_sz (a:size) (b:size) : size =
  match (a, b) with
      (SIZE_fixed a, SIZE_fixed b) -> SIZE_fixed (Int64.mul a b)
    | (a, SIZE_fixed b) -> SIZE_rt_mul (SIZE_fixed b, a)
    | (a, b) -> SIZE_rt_mul (a, b)
;;

let rec max_sz (a:size) (b:size) : size =
  let rec no_negs x =
    match x with
        SIZE_fixed _
      | SIZE_fixup_mem_sz _
      | SIZE_fixup_mem_pos _
      | SIZE_param_size _
      | SIZE_param_align _ -> true
      | SIZE_rt_neg _ -> false
      | SIZE_rt_add (a,b) -> (no_negs a) && (no_negs b)
      | SIZE_rt_mul (a,b) -> (no_negs a) && (no_negs b)
      | SIZE_rt_max (a,b) -> (no_negs a) && (no_negs b)
      | SIZE_rt_align (a,b) -> (no_negs a) && (no_negs b)
  in
    match (a, b) with
        (SIZE_rt_align _, SIZE_fixed 1L) -> a
      | (SIZE_fixed 1L, SIZE_rt_align _) -> b
      | (SIZE_param_align _, SIZE_fixed 1L) -> a
      | (SIZE_fixed 1L, SIZE_param_align _) -> b
      | (a, SIZE_rt_max (b, c)) when a = b -> max_sz a c
      | (a, SIZE_rt_max (b, c)) when a = c -> max_sz a b
      | (SIZE_rt_max (b, c), a) when a = b -> max_sz a c
      | (SIZE_rt_max (b, c), a) when a = c -> max_sz a b
      | (SIZE_fixed a, SIZE_fixed b) -> SIZE_fixed (i64_max a b)
      | (SIZE_fixed 0L, b) when no_negs b -> b
      | (a, SIZE_fixed 0L) when no_negs a -> a
      | (a, SIZE_fixed b) -> max_sz (SIZE_fixed b) a
      | (a, b) when a = b -> a
      | (a, b) -> SIZE_rt_max (a, b)
;;

(* FIXME: audit this carefuly; I am not terribly certain of the
 * algebraic simplification going on here. Sadly, without it
 * the diagnostic output from translation becomes completely
 * illegible.
 *)

let align_sz (a:size) (b:size) : size =
  let rec alignment_of s =
    match s with
        SIZE_rt_align (SIZE_fixed n, s) ->
          let inner_alignment = alignment_of s in
            if (Int64.rem n inner_alignment) = 0L
            then inner_alignment
            else n
      | SIZE_rt_add (SIZE_fixed n, s)
      | SIZE_rt_add (s, SIZE_fixed n) ->
          let inner_alignment = alignment_of s in
            if (Int64.rem n inner_alignment) = 0L
            then inner_alignment
            else 1L (* This could be lcd(...) or such. *)
      | SIZE_rt_max (a, SIZE_fixed 1L) -> alignment_of a
      | SIZE_rt_max (SIZE_fixed 1L, b) -> alignment_of b
      | _ -> 1L
  in
    match (a, b) with
        (SIZE_fixed a, SIZE_fixed b) -> SIZE_fixed (i64_align a b)
      | (SIZE_fixed x, _) when i64_lt x 1L -> bug () "alignment less than 1"
      | (SIZE_fixed 1L, b) -> b (* everything is 1-aligned. *)
      | (_, SIZE_fixed 0L) -> b (* 0 is everything-aligned. *)
      | (SIZE_fixed a, b) ->
          let inner_alignment = alignment_of b in
          if (Int64.rem a inner_alignment) = 0L
          then b
          else SIZE_rt_align (SIZE_fixed a, b)
      | (SIZE_rt_max (a, SIZE_fixed 1L), b) -> SIZE_rt_align (a, b)
      | (SIZE_rt_max (SIZE_fixed 1L, a), b) -> SIZE_rt_align (a, b)
      | (a, b) -> SIZE_rt_align (a, b)
;;

let force_sz (a:size) : int64 =
  match a with
      SIZE_fixed i -> i
    | _ -> bug () "force_sz: forced non-fixed size expression %s"
        (string_of_size a)
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
