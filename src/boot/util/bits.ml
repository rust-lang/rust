type t = {
  storage: int array;
  nbits: int;
}
;;

let int_bits =
  if max_int = (1 lsl 30) - 1
  then 31
  else 63
;;

let create nbits flag =
  { storage = Array.make (nbits / int_bits + 1) (if flag then lnot 0 else 0);
    nbits = nbits }
;;

(* 
 * mutate v0 in place: v0.(i) <- v0.(i) op v1.(i), returning bool indicating
 * whether any bits in v0 changed in the process. 
 *)
let process (op:int -> int -> int) (v0:t) (v1:t) : bool =
  let changed = ref false in
    assert (v0.nbits = v1.nbits);
    assert ((Array.length v0.storage) = (Array.length v1.storage));
    Array.iteri
      begin
        fun i w1 ->
          let w0 = v0.storage.(i) in
          let w0' = op w0 w1 in
            if not (w0' = w0)
            then changed := true;
            v0.storage.(i) <- w0';
      end
      v1.storage;
    !changed
;;

let union = process (lor) ;;
let intersect = process (land) ;;
let copy = process (fun _ w1 -> w1) ;;

let get (v:t) (i:int) : bool =
  assert (i >= 0);
  assert (i < v.nbits);
  let w = i / int_bits in
  let b = i mod int_bits in
  let x = 1 land (v.storage.(w) lsr b) in
    x = 1
;;

let equal (v1:t) (v0:t) : bool =
  v0 = v1
;;

let clear (v:t) : unit =
  for i = 0 to (Array.length v.storage) - 1
  do
    v.storage.(i) <- 0
  done
;;

let invert (v:t) : unit =
  for i = 0 to (Array.length v.storage) - 1
  do
    v.storage.(i) <- lnot v.storage.(i)
  done
;;

(* dst = dst - src *)
let difference (dst:t) (src:t) : bool =
  invert src;
  let b = intersect dst src in
    invert src;
    b
;;


let set (v:t) (i:int) (x:bool) : unit =
  assert (i >= 0);
  assert (i < v.nbits);
  let w = i / int_bits in
  let b = i mod int_bits in
  let w0 = v.storage.(w) in
  let flag = 1 lsl b in
    v.storage.(w) <-
      if x
      then w0 lor flag
      else w0 land (lnot flag)
;;

let to_list (v:t) : int list =
  if v.nbits = 0
  then []
  else
    let accum = ref [] in
    let word = ref v.storage.(0) in
      for i = 0 to (v.nbits-1) do
        if i mod int_bits = 0
        then word := v.storage.(i / int_bits);
        if (1 land (!word)) = 1
        then accum := i :: (!accum);
        word := (!word) lsr 1;
      done;
      !accum
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
