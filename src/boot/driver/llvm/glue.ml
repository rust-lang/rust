(*
 * Glue for the LLVM backend.
 *)

let alt_argspecs sess = [
  ("-llvm", Arg.Unit (fun _ -> sess.Session.sess_alt_backend <- true),
    "emit LLVM bitcode")
];;

let alt_pipeline sess sem_cx crate =
  let process processor =
    processor sem_cx crate;
    if sess.Session.sess_failed then exit 1 else ()
  in
  Array.iter process
    [|
      Resolve.process_crate;
      Type.process_crate;
      Effect.process_crate;
      Typestate.process_crate;
      Loop.process_crate;
      Alias.process_crate;
      Dead.process_crate;
      Layout.process_crate
    |];
  Llemit.trans_and_process_crate sess sem_cx crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

