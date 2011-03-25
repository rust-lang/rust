(*
 * Common formatting helpers.
 *)

let fmt = Format.fprintf
;;

let fmt_str ff = fmt ff "%s"
;;

let fmt_obox ff = Format.pp_open_box ff 4;;
let fmt_obox_n ff n = Format.pp_open_box ff n;;
let fmt_cbox ff = Format.pp_close_box ff ();;
let fmt_obr ff = fmt ff "{";;
let fmt_cbr ff = fmt ff "@\n}";;
let fmt_cbb ff = (fmt_cbox ff; fmt_cbr ff);;
let fmt_break ff = Format.pp_print_space ff ();;

let fmt_bracketed
    (bra:string)
    (ket:string)
    (inner:Format.formatter -> 'a -> unit)
    (ff:Format.formatter)
    (a:'a)
    : unit =
  fmt_str ff bra;
  fmt_obox_n ff 0;
  inner ff a;
  fmt_cbox ff;
  fmt_str ff ket
;;

let fmt_arr_sep
    (sep:string)
    (inner:Format.formatter -> 'a -> unit)
    (ff:Format.formatter)
    (az:'a array)
    : unit =
  Array.iteri
    begin
      fun i a ->
        if i <> 0
        then (fmt_str ff sep; fmt_break ff);
        inner ff a
    end
    az
;;

let fmt_bracketed_arr_sep
    (bra:string)
    (ket:string)
    (sep:string)
    (inner:Format.formatter -> 'a -> unit)
    (ff:Format.formatter)
    (az:'a array)
    : unit =
  fmt_bracketed bra ket
    (fmt_arr_sep sep inner)
    ff az
;;

let fmt_to_str (f:Format.formatter -> 'a -> unit) (v:'a) : string =
  let buf = Buffer.create 16 in
  let bf = Format.formatter_of_buffer buf in
    begin
      f bf v;
      Format.pp_print_flush bf ();
      Buffer.contents buf
    end
;;

let sprintf_fmt
    (f:Format.formatter -> 'a -> unit)
    : (unit -> 'a -> string) =
  (fun _ -> fmt_to_str f)
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
