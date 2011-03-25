(*

   Module for writing Microsoft PE files

   Every image has a base address it's to be loaded at.

   "file pointer" = offset in file

   "VA" = address at runtime

   "RVA" = VA - base address

   If you write a non-RVA absolute address at any point you must put it
   in a rebasing list so the loader can adjust it when/if it has to load
   you at a different address.

   Almost all addresses in the file are RVAs. Worry about the VAs.

*)

open Asm;;
open Common;;

let log (sess:Session.sess) =
  Session.log "obj (pe)"
    sess.Session.sess_log_obj
    sess.Session.sess_log_out
;;

let iflog (sess:Session.sess) (thunk:(unit -> unit)) : unit =
  if sess.Session.sess_log_obj
  then thunk ()
  else ()
;;

(*

   The default image base (VA) for an executable on Win32 is 0x400000.

   We use this too. RVAs are relative to this. RVA 0 = VA 0x400000.

   Alignments are also relatively standard and fixed for Win32/PE32:
   4k memory pages, 512 byte disk sectors.

   Since this is a stupid emitter, and we're not generating an awful
   lot of sections, we are not going to differentiate between these
   two kinds of alignment: we just align our sections to memory pages
   and sometimes waste most of them. Shucks.

*)

let pe_image_base = 0x400000L;;
let pe_file_alignment = 0x200;;
let pe_mem_alignment = 0x1000;;

let rva (f:fixup) = (SUB ((M_POS f), (IMM pe_image_base)));;

let def_file_aligned f i =
  ALIGN_FILE
    (pe_file_alignment,
     SEQ [|
       DEF(f,
           SEQ [| i;
                  ALIGN_FILE
                    (pe_file_alignment, MARK) |]) |] )
;;

let def_mem_aligned f i =
  ALIGN_MEM
    (pe_mem_alignment,
     SEQ [|
       DEF(f,
           SEQ [| i;
                  ALIGN_MEM
                    (pe_mem_alignment, MARK) |]) |] )
;;

let align_both i =
  ALIGN_FILE (pe_file_alignment,
              (ALIGN_MEM (pe_mem_alignment, i)))
;;

let def_aligned f i =
  align_both
    (SEQ [| DEF(f,i);
            (align_both MARK)|])
;;


(*

  At the beginning of a PE file there is an MS-DOS stub, 0x00 - 0x7F,
  that we just insert literally. It prints "This program must be run
  under Win32" and exits. Woo!

  Within it, at offset 0x3C, there is an encoded offset of the PE
  header we actually care about. So 0x3C - 0x3F are 0x00000100 (LE)
  which say "the PE header is actually at 0x100", a nice sensible spot
  for it. We pad the next 128 bytes out to 0x100 and start there for
  real.

  From then on in it's a sensible object file. Here's the MS-DOS bit.
*)

let pe_msdos_header_and_padding
    : frag =
  SEQ [|
    BYTES
      [|
        (* 00000000 *)
        0x4d; 0x5a; 0x50; 0x00; 0x02; 0x00; 0x00; 0x00;
        0x04; 0x00; 0x0f; 0x00; 0xff; 0xff; 0x00; 0x00;

        (* 00000010 *)
        0xb8; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00;
        0x40; 0x00; 0x1a; 0x00; 0x00; 0x00; 0x00; 0x00;

        (* 00000020 *)
        0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00;
        0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00;

        (* 00000030 *)
        0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00;
        0x00; 0x00; 0x00; 0x00; 0x00; 0x01; 0x00; 0x00;
        (*                      ^^^^PE HDR offset^^^^^ *)

        (* 00000040 *)
        0xba; 0x10; 0x00; 0x0e; 0x1f; 0xb4; 0x09; 0xcd;
        0x21; 0xb8; 0x01; 0x4c; 0xcd; 0x21; 0x90; 0x90;

        (* 00000050 *)
        0x54; 0x68; 0x69; 0x73; 0x20; 0x70; 0x72; 0x6f;  (* "This pro" *)
        0x67; 0x72; 0x61; 0x6d; 0x20; 0x6d; 0x75; 0x73;  (* "gram mus" *)

        (* 00000060 *)
        0x74; 0x20; 0x62; 0x65; 0x20; 0x72; 0x75; 0x6e;  (* "t be run" *)
        0x20; 0x75; 0x6e; 0x64; 0x65; 0x72; 0x20; 0x57;  (* " under W" *)

        (* 00000070 *)
        0x69; 0x6e; 0x33; 0x32; 0x0d; 0x0a; 0x24; 0x37;  (* "in32\r\n" *)
        0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00; 0x00;
      |];
    PAD 0x80
  |]
;;

(*
  A work of art, is it not? Take a moment to appreciate the madness.

  All done? Ok, now on to the PE header proper.

  PE headers are just COFF headers with a little preamble.
*)

type pe_machine =
    (* Maybe support more later. *)
    IMAGE_FILE_MACHINE_AMD64
  | IMAGE_FILE_MACHINE_I386
;;


let pe_timestamp _ =
  Int64.of_float (Unix.gettimeofday())
;;


type pe_characteristics =
    (* Maybe support more later. *)
    IMAGE_FILE_RELOCS_STRIPPED
  | IMAGE_FILE_EXECUTABLE_IMAGE
  | IMAGE_FILE_LINE_NUMS_STRIPPED
  | IMAGE_FILE_LOCAL_SYMS_STRIPPED
  | IMAGE_FILE_32BIT_MACHINE
  | IMAGE_FILE_DEBUG_STRIPPED
  | IMAGE_FILE_DLL
;;


let pe_header
    ~(machine:pe_machine)
    ~(symbol_table_fixup:fixup)
    ~(number_of_sections:int64)
    ~(number_of_symbols:int64)
    ~(loader_hdr_fixup:fixup)
    ~(characteristics:pe_characteristics list)
    : frag =
  ALIGN_FILE
    (8,
     SEQ [|
       STRING "PE\x00\x00";
       WORD (TY_u16, (IMM (match machine with
                               IMAGE_FILE_MACHINE_AMD64 -> 0x8664L
                             | IMAGE_FILE_MACHINE_I386 -> 0x014cL)));
       WORD (TY_u16, (IMM number_of_sections));
       WORD (TY_u32, (IMM (pe_timestamp())));
       WORD (TY_u32, (F_POS symbol_table_fixup));
       WORD (TY_u32, (IMM number_of_symbols));
       WORD (TY_u16, (F_SZ loader_hdr_fixup));
       WORD (TY_u16, (IMM (fold_flags
                      (fun c -> match c with
                           IMAGE_FILE_RELOCS_STRIPPED -> 0x1L
                         | IMAGE_FILE_EXECUTABLE_IMAGE -> 0x2L
                         | IMAGE_FILE_LINE_NUMS_STRIPPED -> 0x4L
                         | IMAGE_FILE_LOCAL_SYMS_STRIPPED -> 0x8L
                         | IMAGE_FILE_32BIT_MACHINE -> 0x100L
                         | IMAGE_FILE_DEBUG_STRIPPED -> 0x200L
                         | IMAGE_FILE_DLL -> 0x2000L)
                      characteristics)))
     |])
;;

(*

   After the PE header comes an "optional" header for the loader. In
   our case this is hardly optional since we are producing a file for
   the loader.

*)

type pe_subsystem =
    (* Maybe support more later. *)
    IMAGE_SUBSYSTEM_WINDOWS_GUI
  | IMAGE_SUBSYSTEM_WINDOWS_CUI
;;

let zero32 = WORD (TY_u32, (IMM 0L))
;;

let pe_loader_header
    ~(text_fixup:fixup)
    ~(init_data_fixup:fixup)
    ~(size_of_uninit_data:int64)
    ~(entry_point_fixup:fixup option)
    ~(image_fixup:fixup)
    ~(all_hdrs_fixup:fixup)
    ~(subsys:pe_subsystem)
    ~(loader_hdr_fixup:fixup)
    ~(import_dir_fixup:fixup)
    ~(export_dir_fixup:fixup)
    : frag =
  DEF
    (loader_hdr_fixup,
     SEQ [|
       WORD (TY_u16, (IMM 0x10bL));          (* COFF magic tag for PE32.  *)
       (* Snagged *)
       WORD (TY_u8, (IMM 0x2L));             (* Linker major version.     *)
       WORD (TY_u8, (IMM 0x38L));            (* Linker minor version.     *)

       WORD (TY_u32, (F_SZ text_fixup));     (* "size of code"            *)
       WORD (TY_u32,                         (* "size of all init data"   *)
             (F_SZ init_data_fixup));
       WORD (TY_u32,
             (IMM size_of_uninit_data));

       begin
         match entry_point_fixup with
             None -> zero32                  (* Library mode: DLLMain     *)
           | Some entry_point_fixup ->
               WORD (TY_u32,
                     (rva
                        entry_point_fixup))  (* "address of entry point"  *)
       end;

       WORD (TY_u32, (rva text_fixup));      (* "base of code"            *)
       WORD (TY_u32, (rva init_data_fixup)); (* "base of data"            *)
       WORD (TY_u32, (IMM pe_image_base));
       WORD (TY_u32, (IMM (Int64.of_int
                      pe_mem_alignment)));
       WORD (TY_u32, (IMM (Int64.of_int
                      pe_file_alignment)));

       WORD (TY_u16, (IMM 4L));             (* Major OS version: NT4.     *)
       WORD (TY_u16, (IMM 0L));             (* Minor OS version.          *)
       WORD (TY_u16, (IMM 1L));             (* Major image version.       *)
       WORD (TY_u16, (IMM 0L));             (* Minor image version.       *)
       WORD (TY_u16, (IMM 4L));             (* Major subsystem version.   *)
       WORD (TY_u16, (IMM 0L));             (* Minor subsystem version.   *)

       zero32;                              (* Reserved.                  *)

       WORD (TY_u32, (M_SZ image_fixup));
       WORD (TY_u32, (M_SZ all_hdrs_fixup));

       zero32;                              (* Checksum, but OK if zero.  *)
       WORD (TY_u16, (IMM (match subsys with
                        IMAGE_SUBSYSTEM_WINDOWS_GUI -> 2L
                      | IMAGE_SUBSYSTEM_WINDOWS_CUI -> 3L)));

       WORD (TY_u16, (IMM 0L));             (* DLL characteristics.       *)

       WORD (TY_u32, (IMM 0x100000L));      (* Size of stack reserve.     *)
       WORD (TY_u32, (IMM 0x4000L));        (* Size of stack commit.      *)

       WORD (TY_u32, (IMM 0x100000L));      (* Size of heap reserve.      *)
       WORD (TY_u32, (IMM 0x1000L));        (* Size of heap commit.       *)

       zero32;                              (* Reserved.                  *)
       WORD (TY_u32, (IMM 16L));            (* Number of dir references.  *)

       (* Begin directories, variable part of hdr.        *)

       (*

         Standard PE files have ~10 directories referenced from
         here. We only fill in two of them -- the export/import
         directories -- because we don't care about the others. We
         leave the rest as zero in case someone is looking for
         them. This may be superfluous or wrong.

       *)


       WORD (TY_u32, (rva export_dir_fixup));
       WORD (TY_u32, (M_SZ export_dir_fixup));

       WORD (TY_u32, (rva import_dir_fixup));
       WORD (TY_u32, (M_SZ import_dir_fixup));

       zero32; zero32;    (* Resource dir.      *)
       zero32; zero32;    (* Exception dir.     *)
       zero32; zero32;    (* Security dir.      *)
       zero32; zero32;    (* Base reloc dir.    *)
       zero32; zero32;    (* Debug dir.         *)
       zero32; zero32;    (* Image desc dir.    *)
       zero32; zero32;    (* Mach spec dir.     *)
       zero32; zero32;    (* TLS dir.           *)

       zero32; zero32;    (* Load config.       *)
       zero32; zero32;    (* Bound import.      *)
       zero32; zero32;    (* IAT                *)
       zero32; zero32;    (* Delay import.      *)
       zero32; zero32;    (* COM descriptor     *)
       zero32; zero32;    (* ????????           *)
     |])

;;


type pe_section_id =
    (* Maybe support more later. *)
    SECTION_ID_TEXT
  | SECTION_ID_DATA
  | SECTION_ID_RDATA
  | SECTION_ID_BSS
  | SECTION_ID_IMPORTS
  | SECTION_ID_EXPORTS
  | SECTION_ID_DEBUG_ARANGES
  | SECTION_ID_DEBUG_PUBNAMES
  | SECTION_ID_DEBUG_INFO
  | SECTION_ID_DEBUG_ABBREV
  | SECTION_ID_DEBUG_LINE
  | SECTION_ID_DEBUG_FRAME
  | SECTION_ID_NOTE_RUST
;;

type pe_section_characteristics =
    (* Maybe support more later. *)
    IMAGE_SCN_CNT_CODE
  | IMAGE_SCN_CNT_INITIALIZED_DATA
  | IMAGE_SCN_CNT_UNINITIALIZED_DATA
  | IMAGE_SCN_MEM_DISCARDABLE
  | IMAGE_SCN_MEM_SHARED
  | IMAGE_SCN_MEM_EXECUTE
  | IMAGE_SCN_MEM_READ
  | IMAGE_SCN_MEM_WRITE

let pe_section_header
    ~(id:pe_section_id)
    ~(hdr_fixup:fixup)
    : frag =
  let
      characteristics =
    match id with
        SECTION_ID_TEXT -> [ IMAGE_SCN_CNT_CODE;
                             IMAGE_SCN_MEM_READ;
                             IMAGE_SCN_MEM_EXECUTE ]
      | SECTION_ID_DATA -> [ IMAGE_SCN_CNT_INITIALIZED_DATA;
                             IMAGE_SCN_MEM_READ;
                             IMAGE_SCN_MEM_WRITE ]
      | SECTION_ID_BSS -> [ IMAGE_SCN_CNT_UNINITIALIZED_DATA;
                            IMAGE_SCN_MEM_READ;
                            IMAGE_SCN_MEM_WRITE ]
      | SECTION_ID_IMPORTS -> [ IMAGE_SCN_CNT_INITIALIZED_DATA;
                                IMAGE_SCN_MEM_READ;
                                IMAGE_SCN_MEM_WRITE ]
      | SECTION_ID_EXPORTS -> [ IMAGE_SCN_CNT_INITIALIZED_DATA;
                                IMAGE_SCN_MEM_READ ]
      | SECTION_ID_RDATA
      | SECTION_ID_DEBUG_ARANGES
      | SECTION_ID_DEBUG_PUBNAMES
      | SECTION_ID_DEBUG_INFO
      | SECTION_ID_DEBUG_ABBREV
      | SECTION_ID_DEBUG_LINE
      | SECTION_ID_DEBUG_FRAME
      | SECTION_ID_NOTE_RUST -> [ IMAGE_SCN_CNT_INITIALIZED_DATA;
                                  IMAGE_SCN_MEM_READ ]
  in
    SEQ [|
      STRING
        begin
          match id with
              SECTION_ID_TEXT -> ".text\x00\x00\x00"
            | SECTION_ID_DATA -> ".data\x00\x00\x00"
            | SECTION_ID_RDATA -> ".rdata\x00\x00"
            | SECTION_ID_BSS -> ".bss\x00\x00\x00\x00"
            | SECTION_ID_IMPORTS -> ".idata\x00\x00"
            | SECTION_ID_EXPORTS -> ".edata\x00\x00"

            (* There is a bizarre Microsoft COFF extension to account
             * for longer-than-8-char section names: you emit a single
             * '/' character then the ASCII-numeric encoding of the
             * offset within the file's string table of the full name.
             * So we put all our extended section names at the
             * beginning of the string table in a very specific order
             * and hard-wire the offsets as "names" here. You could
             * theoretically extend this to a "new kind" of fixup
             * reference (ASCII_POS or such), if you feel this is
             * something you want to twiddle with.
             *)

            | SECTION_ID_DEBUG_ARANGES  -> "/4\x00\x00\x00\x00\x00\x00"
            | SECTION_ID_DEBUG_PUBNAMES -> "/19\x00\x00\x00\x00\x00"
            | SECTION_ID_DEBUG_INFO     -> "/35\x00\x00\x00\x00\x00"
            | SECTION_ID_DEBUG_ABBREV   -> "/47\x00\x00\x00\x00\x00"
            | SECTION_ID_DEBUG_LINE     -> "/61\x00\x00\x00\x00\x00"
            | SECTION_ID_DEBUG_FRAME    -> "/73\x00\x00\x00\x00\x00"
            | SECTION_ID_NOTE_RUST      -> "/86\x00\x00\x00\x00\x00"
        end;

      (* The next two pairs are only supposed to be different if the
         file and section alignments differ. This is a stupid emitter
         so they're not, no problem. *)

      WORD (TY_u32, (M_SZ hdr_fixup));  (* "Virtual size"    *)
      WORD (TY_u32, (rva hdr_fixup));   (* "Virtual address" *)

      WORD (TY_u32, (F_SZ hdr_fixup));  (* "Size of raw data"    *)
      WORD (TY_u32, (F_POS hdr_fixup)); (* "Pointer to raw data" *)

      zero32;      (* Reserved. *)
      zero32;      (* Reserved. *)
      zero32;      (* Reserved. *)

      WORD (TY_u32, (IMM (fold_flags
                     (fun c -> match c with
                          IMAGE_SCN_CNT_CODE -> 0x20L
                        | IMAGE_SCN_CNT_INITIALIZED_DATA -> 0x40L
                        | IMAGE_SCN_CNT_UNINITIALIZED_DATA -> 0x80L
                        | IMAGE_SCN_MEM_DISCARDABLE -> 0x2000000L
                        | IMAGE_SCN_MEM_SHARED -> 0x10000000L
                        | IMAGE_SCN_MEM_EXECUTE -> 0x20000000L
                        | IMAGE_SCN_MEM_READ -> 0x40000000L
                        | IMAGE_SCN_MEM_WRITE -> 0x80000000L)
                     characteristics)))
    |]
;;


(*

   "Thunk" is a misnomer here; the thunk RVA is the address of a word
   that the loader will store an address into. The stored address is
   the address of the imported object.

   So if the imported object is X, and the thunk slot is Y, the loader
   is doing "Y = &X" and returning &Y as the thunk RVA. To load datum X
   after the imports are resolved, given the thunk RVA R, you load
   **R.

*)

type pe_import =
    {
      pe_import_name_fixup: fixup;
      pe_import_name: string;
      pe_import_address_fixup: fixup;
    }

type pe_import_dll_entry =
    {
      pe_import_dll_name_fixup: fixup;
      pe_import_dll_name: string;
      pe_import_dll_ILT_fixup: fixup;
      pe_import_dll_IAT_fixup: fixup;
      pe_import_dll_imports: pe_import array;
    }

  (*

     The import section .idata has a mostly self-contained table
     structure. You feed it a list of DLL entries, each of which names
     a DLL and lists symbols in the DLL to import.

     For each named symbol, a 4-byte slot will be reserved in an
     "import lookup table" (ILT, also in this section). The slot is
     a pointer to a string in this section giving the name.

     Immediately *after* the ILT, there is an "import address table" (IAT),
     which is initially identical to the ILT. The loader replaces the entries
     in the IAT slots with the imported pointers at runtime.

     A central directory at the start of the section lists all the the import
     thunk tables. Each entry in the import directory is 20 bytes (5 words)
     but only the last 2 are used: the second last is a pointer to the string
     name of the DLL in question (string also in this section) and the last is
     a pointer to the import thunk table itself (also in this section).

     Curiously, of the 5 documents I've consulted on the nature of the
     first 3 fields, I find a variety of interpretations.

  *)

let pe_import_section
    ~(import_dir_fixup:fixup)
    ~(dlls:pe_import_dll_entry array)
    : frag =

  let form_dir_entry
      (entry:pe_import_dll_entry)
      : frag =
    SEQ [|
      (* Note: documented opinions vary greatly about whether the
         first, last, or both of the slots in one of these rows points
         to the RVA of the name/hint used to look the import up. This
         table format is a mess!  *)
      WORD (TY_u32,
            (rva
               entry.pe_import_dll_ILT_fixup)); (* Import lookup table. *)
      WORD (TY_u32, (IMM 0L));                  (* Timestamp, unused.   *)
      WORD (TY_u32, (IMM 0x0L));                (* Forwarder chain, unused. *)
      WORD (TY_u32, (rva entry.pe_import_dll_name_fixup));
      WORD (TY_u32,
            (rva
               entry.pe_import_dll_IAT_fixup)); (* Import address table.*)
    |]
  in

  let form_ILT_slot
      (import:pe_import)
      : frag =
    (WORD (TY_u32, (rva import.pe_import_name_fixup)))
  in

  let form_IAT_slot
      (import:pe_import)
      : frag =
    (DEF (import.pe_import_address_fixup,
          (WORD (TY_u32, (rva import.pe_import_name_fixup)))))
  in

  let form_tables_for_dll
      (dll:pe_import_dll_entry)
      : frag =
    let terminator = WORD (TY_u32, (IMM 0L)) in
    let ilt =
      SEQ [|
        SEQ (Array.map form_ILT_slot dll.pe_import_dll_imports);
        terminator
      |]
    in
    let iat =
      SEQ [|
        SEQ (Array.map form_IAT_slot dll.pe_import_dll_imports);
        terminator
      |]
    in
      if Array.length dll.pe_import_dll_imports < 1
      then bug () "Pe.form_tables_for_dll: empty imports"
      else
        SEQ [|
          DEF (dll.pe_import_dll_ILT_fixup, ilt);
          DEF (dll.pe_import_dll_IAT_fixup, iat)
        |]

  in

  let form_import_string
      (import:pe_import)
      : frag =
    DEF
      (import.pe_import_name_fixup,
       SEQ [|
         (* import string entries begin with a 2-byte "hint", but we just
            set it to zero.  *)
         (WORD (TY_u16, (IMM 0L)));
         ZSTRING import.pe_import_name;
         (if String.length import.pe_import_name mod 2 == 0
          then PAD 1
          else PAD 0)
       |])
  in

  let form_dir_entry_string
      (dll:pe_import_dll_entry)
      : frag =
    DEF
      (dll.pe_import_dll_name_fixup,
       SEQ [| ZSTRING dll.pe_import_dll_name;
              (if String.length dll.pe_import_dll_name mod 2 == 0
               then PAD 1
               else PAD 0);
              SEQ (Array.map form_import_string dll.pe_import_dll_imports) |])
  in

  let dir = SEQ (Array.map form_dir_entry dlls) in
  let dir_terminator = PAD 20 in
  let tables = SEQ (Array.map form_tables_for_dll dlls) in
  let strings = SEQ (Array.map form_dir_entry_string dlls)
  in
    def_aligned
      import_dir_fixup
      (SEQ
         [|
           dir;
           dir_terminator;
           tables;
           strings
         |])

;;

type pe_export =
    {
      pe_export_name_fixup: fixup;
      pe_export_name: string;
      pe_export_address_fixup: fixup;
    }
;;

let pe_export_section
    ~(sess:Session.sess)
    ~(export_dir_fixup:fixup)
    ~(exports:pe_export array)
    : frag =
  Array.sort (fun a b -> compare a.pe_export_name b.pe_export_name) exports;
  let export_addr_table_fixup = new_fixup "export address table" in
  let export_addr_table =
    DEF
      (export_addr_table_fixup,
       SEQ
         (Array.map
            (fun e -> (WORD (TY_u32, rva e.pe_export_address_fixup)))
            exports))
  in
  let export_name_pointer_table_fixup =
      new_fixup "export name pointer table"
  in
  let export_name_pointer_table =
    DEF
      (export_name_pointer_table_fixup,
       SEQ
         (Array.map
            (fun e -> (WORD (TY_u32, rva e.pe_export_name_fixup)))
            exports))
  in
  let export_name_table_fixup = new_fixup "export name table" in
  let export_name_table =
    DEF
      (export_name_table_fixup,
       SEQ
         (Array.map
            (fun e -> (DEF (e.pe_export_name_fixup,
                            (ZSTRING e.pe_export_name))))
            exports))
  in
  let export_ordinal_table_fixup = new_fixup "export ordinal table" in
  let export_ordinal_table =
    DEF
      (export_ordinal_table_fixup,
       SEQ
         (Array.mapi
            (fun i _ -> (WORD (TY_u16, IMM (Int64.of_int (i)))))
            exports))
  in
  let image_name_fixup = new_fixup "image name fixup" in
  let n_exports = IMM (Int64.of_int (Array.length exports)) in
  let export_dir_table =
    SEQ [|
      WORD (TY_u32, IMM 0L);               (* Flags, reserved.    *)
      WORD (TY_u32, IMM 0L);               (* Timestamp, unused.  *)
      WORD (TY_u16, IMM 0L);               (* Major vers., unused *)
      WORD (TY_u16, IMM 0L);               (* Minor vers., unused *)
      WORD (TY_u32, rva image_name_fixup); (* Name RVA.           *)
      WORD (TY_u32, IMM 1L);               (* Ordinal base = 1.   *)
      WORD (TY_u32, n_exports);          (* # entries in EAT.     *)
      WORD (TY_u32, n_exports);          (* # entries in ENPT/EOT.*)
      WORD (TY_u32, rva export_addr_table_fixup);         (* EAT  *)
      WORD (TY_u32, rva export_name_pointer_table_fixup); (* ENPT *)
      WORD (TY_u32, rva export_ordinal_table_fixup);      (* EOT  *)
    |]
  in
    def_aligned export_dir_fixup
      (SEQ [|
         export_dir_table;
         export_addr_table;
         export_name_pointer_table;
         export_ordinal_table;
         DEF (image_name_fixup,
              ZSTRING (Session.filename_of sess.Session.sess_out));
         export_name_table
       |])
;;

let pe_text_section
    ~(sess:Session.sess)
    ~(sem:Semant.ctxt)
    ~(start_fixup:fixup option)
    ~(rust_start_fixup:fixup option)
    ~(main_fn_fixup:fixup option)
    ~(text_fixup:fixup)
    ~(crate_code:frag)
    : frag =
  let startup =
    match (start_fixup, rust_start_fixup, main_fn_fixup) with
        (None, _, _)
      | (_, None, _)
      | (_, _, None) -> MARK
      | (Some start_fixup,
         Some rust_start_fixup,
         Some main_fn_fixup) ->
          let e = X86.new_emitter_without_vregs () in
            (*
             * We are called from the Microsoft C library startup routine,
             * and assumed to be stdcall; so we have to clean up our own
             * stack before returning.
             *)
            X86.objfile_start e
              ~start_fixup
              ~rust_start_fixup
              ~main_fn_fixup
              ~crate_fixup: sem.Semant.ctxt_crate_fixup
              ~indirect_start: true;
            X86.frags_of_emitted_quads sess e;
  in
    def_aligned
      text_fixup
      (SEQ [|
         startup;
         crate_code
       |])
;;

let rustrt_imports sem =
  let make_imports_for_lib (lib, tab) =
    {
      pe_import_dll_name_fixup = new_fixup "dll name";
      pe_import_dll_name = (match lib with
                                REQUIRED_LIB_rustrt -> "rustrt.dll"
                              | REQUIRED_LIB_crt -> "msvcrt.dll"
                              | REQUIRED_LIB_rust ls
                              | REQUIRED_LIB_c ls -> ls.required_libname);
      pe_import_dll_ILT_fixup = new_fixup "dll ILT";
      pe_import_dll_IAT_fixup = new_fixup "dll IAT";
      pe_import_dll_imports =
        Array.of_list
          (List.map
             begin
               fun (name, fixup) ->
                 {
                   pe_import_name_fixup = new_fixup "import name";
                   pe_import_name = name;
                   pe_import_address_fixup = fixup;
                 }
             end
             (htab_pairs tab))
    }
  in
    Array.of_list
      (List.map
         make_imports_for_lib
         (htab_pairs sem.Semant.ctxt_native_required))
;;


let crate_exports (sem:Semant.ctxt) : pe_export array =
  let export_sym (name, fixup) =
    {
      pe_export_name_fixup = new_fixup "export name fixup";
      pe_export_name = name;
      pe_export_address_fixup = fixup;
    }
  in
  let export_seg (_, tab) =
    Array.of_list (List.map export_sym (htab_pairs tab))
  in

  (* Make some fake symbol table entries to aid in debugging. *)
  let export_stab name fixup =
    {
      pe_export_name_fixup = new_fixup "export name fixup";
      pe_export_name = "rust$" ^ name;
      pe_export_address_fixup = fixup
    }
  in
  let export_stab_of_item (node_id, code) =
    let name = Hashtbl.find sem.Semant.ctxt_all_item_names node_id in
    let name' = "item$" ^ (Semant.string_of_name name) in
    export_stab name' code.Semant.code_fixup
  in
  let export_stab_of_glue (glue, code) =
    export_stab (Semant.glue_str sem glue) code.Semant.code_fixup
  in

  let stabs =
    Array.of_list (List.concat [
      (List.map export_stab_of_item
        (htab_pairs sem.Semant.ctxt_all_item_code));
      (List.map export_stab_of_glue (htab_pairs sem.Semant.ctxt_glue_code))
    ])
  in

    Array.concat
      (stabs::(List.map export_seg
         (htab_pairs sem.Semant.ctxt_native_provided)))
;;


let emit_file
    (sess:Session.sess)
    (crate:Ast.crate)
    (code:Asm.frag)
    (data:Asm.frag)
    (sem:Semant.ctxt)
    (dw:Dwarf.debug_records)
    : unit =

  let all_hdrs_fixup = new_fixup "all headers" in
  let all_init_data_fixup = new_fixup "all initialized data" in
  let loader_hdr_fixup = new_fixup "loader header" in
  let import_dir_fixup = new_fixup "import directory" in
  let export_dir_fixup = new_fixup "export directory" in
  let text_fixup = new_fixup "text section" in
  let bss_fixup = new_fixup "bss section" in
  let data_fixup = new_fixup "data section" in
  let image_fixup = new_fixup "image fixup" in
  let symtab_fixup = new_fixup "symbol table" in
  let strtab_fixup = new_fixup "string table" in
  let note_rust_fixup = new_fixup ".note.rust section" in

  let (start_fixup, rust_start_fixup) =
    if sess.Session.sess_library_mode
    then (None, None)
    else
      (Some (new_fixup "start"),
       Some (Semant.require_native sem REQUIRED_LIB_rustrt "rust_start"))
  in

  let header = (pe_header
                  ~machine: IMAGE_FILE_MACHINE_I386
                  ~symbol_table_fixup: symtab_fixup
                  ~number_of_sections: 8L
                  ~number_of_symbols: 0L
                  ~loader_hdr_fixup: loader_hdr_fixup
                  ~characteristics:([IMAGE_FILE_EXECUTABLE_IMAGE;
                                    IMAGE_FILE_LINE_NUMS_STRIPPED;
                                    IMAGE_FILE_32BIT_MACHINE;]
                                    @
                                    (if sess.Session.sess_library_mode
                                     then [ IMAGE_FILE_DLL ]
                                     else [ ])))
  in
  let symtab =
    (* 
     * We're not actually presenting a "symbol table", but wish to
     * provide a "string table" which comes immediately *after* the
     * symbol table. It's a violation of the PE spec to put one of
     * these in an executable file (as opposed to just loadable) but
     * it's necessary to communicate the debug section names to GDB,
     * and nobody else complains.  
     *)
    (def_aligned
       symtab_fixup
       (def_aligned
          strtab_fixup
          (SEQ
             [|
               WORD (TY_u32, (F_SZ strtab_fixup));
               ZSTRING ".debug_aranges";
               ZSTRING ".debug_pubnames";
               ZSTRING ".debug_info";
               ZSTRING ".debug_abbrev";
               ZSTRING ".debug_line";
               ZSTRING ".debug_frame";
               ZSTRING ".note.rust";
             |])))
  in
  let loader_header = (pe_loader_header
                         ~text_fixup
                         ~init_data_fixup: all_init_data_fixup
                         ~size_of_uninit_data: 0L
                         ~entry_point_fixup: start_fixup
                         ~image_fixup: image_fixup
                         ~subsys: IMAGE_SUBSYSTEM_WINDOWS_CUI
                         ~all_hdrs_fixup
                         ~loader_hdr_fixup
                         ~import_dir_fixup
                         ~export_dir_fixup)
  in
  let text_header = (pe_section_header
                       ~id: SECTION_ID_TEXT
                       ~hdr_fixup: text_fixup)

  in
  let bss_header = (pe_section_header
                      ~id: SECTION_ID_BSS
                      ~hdr_fixup: bss_fixup)
  in
  let import_section = (pe_import_section
                          ~import_dir_fixup
                          ~dlls: (rustrt_imports sem))
  in
  let import_header = (pe_section_header
                         ~id: SECTION_ID_IMPORTS
                         ~hdr_fixup: import_dir_fixup)
  in
  let export_section = (pe_export_section
                          ~sess
                          ~export_dir_fixup
                          ~exports: (crate_exports sem))
  in
  let export_header = (pe_section_header
                         ~id: SECTION_ID_EXPORTS
                         ~hdr_fixup: export_dir_fixup)
  in
  let data_header = (pe_section_header
                       ~id: SECTION_ID_DATA
                       ~hdr_fixup: data_fixup)
  in
(*
  let debug_aranges_header =
    (pe_section_header
      ~id: SECTION_ID_DEBUG_ARANGES
      ~hdr_fixup: sem.Semant.ctxt_debug_aranges_fixup)
  in
  let debug_pubnames_header =
    (pe_section_header
      ~id: SECTION_ID_DEBUG_PUBNAMES
      ~hdr_fixup: sem.Semant.ctxt_debug_pubnames_fixup)
  in
*)
  let debug_info_header = (pe_section_header
                             ~id: SECTION_ID_DEBUG_INFO
                             ~hdr_fixup: sem.Semant.ctxt_debug_info_fixup)
  in
  let debug_abbrev_header = (pe_section_header
                               ~id: SECTION_ID_DEBUG_ABBREV
                               ~hdr_fixup: sem.Semant.ctxt_debug_abbrev_fixup)
  in
(*
  let debug_line_header =
    (pe_section_header
      ~id: SECTION_ID_DEBUG_LINE
      ~hdr_fixup: sem.Semant.ctxt_debug_line_fixup)
  in
  let debug_frame_header =
    (pe_section_header
      ~id: SECTION_ID_DEBUG_FRAME
      ~hdr_fixup: sem.Semant.ctxt_debug_frame_fixup)
  in
*)
  let note_rust_header = (pe_section_header
                            ~id: SECTION_ID_NOTE_RUST
                            ~hdr_fixup: note_rust_fixup)
  in
  let all_headers = (def_file_aligned
                       all_hdrs_fixup
                       (SEQ
                          [|
                            pe_msdos_header_and_padding;
                            header;
                            loader_header;
                            text_header;
                            bss_header;
                            import_header;
                            export_header;
                            data_header;
                            (*
                            debug_aranges_header;
                            debug_pubnames_header;
                            *)
                            debug_info_header;
                            debug_abbrev_header;
                            (*
                            debug_line_header;
                            debug_frame_header;
                            *)
                            note_rust_header;
                          |]))
  in

  let text_section = (pe_text_section
                        ~sem
                        ~sess
                        ~start_fixup
                        ~rust_start_fixup
                        ~main_fn_fixup: sem.Semant.ctxt_main_fn_fixup
                        ~text_fixup
                        ~crate_code: code)
  in
  let bss_section = def_aligned bss_fixup (BSS 0x10L)
  in
  let data_section = (def_aligned data_fixup
                        (SEQ [| data; symtab; |]))
  in
  let all_init_data = (def_aligned
                         all_init_data_fixup
                         (SEQ [| import_section;
                                 export_section;
                                 data_section; |]))
  in
(*
  let debug_aranges_section =
    def_aligned sem.Semant.ctxt_debug_aranges_fixup dw.Dwarf.debug_aranges
  in
  let debug_pubnames_section =
    def_aligned sem.Semant.ctxt_debug_pubnames_fixup dw.Dwarf.debug_pubnames
  in
*)
  let debug_info_section =
    def_aligned sem.Semant.ctxt_debug_info_fixup dw.Dwarf.debug_info
  in
  let debug_abbrev_section =
    def_aligned sem.Semant.ctxt_debug_abbrev_fixup dw.Dwarf.debug_abbrev
  in
(*
  let debug_line_section =
    def_aligned sem.Semant.ctxt_debug_line_fixup dw.Dwarf.debug_line
  in
  let debug_frame_section =
    def_aligned sem.Semant.ctxt_debug_frame_fixup dw.Dwarf.debug_frame
  in
*)
  let note_rust_section =
    def_aligned note_rust_fixup
      (Asm.note_rust_frags crate.node.Ast.crate_meta)
  in

  let all_frags = SEQ [| MEMPOS pe_image_base;
                         (def_file_aligned image_fixup
                            (SEQ [| DEF (sem.Semant.ctxt_image_base_fixup,
                                         MARK);
                                    all_headers;
                                    text_section;
                                    bss_section;
                                    all_init_data;
                                    (* debug_aranges_section; *)
                                    (* debug_pubnames_section; *)
                                    debug_info_section;
                                    debug_abbrev_section;
                                    (* debug_line_section; *)
                                    (* debug_frame_section; *)
                                    note_rust_section;
                                    ALIGN_MEM (pe_mem_alignment, MARK)
                                 |]
                            )
                         )
                      |]
  in
    write_out_frag sess true all_frags
;;

let pe_magic = "PE";;

let sniff
    (sess:Session.sess)
    (filename:filename)
    : asm_reader option =
  try
    let stat = Unix.stat filename in
    if (stat.Unix.st_kind = Unix.S_REG) &&
      (stat.Unix.st_size >= pe_file_alignment)
    then
      let ar = new_asm_reader sess filename in
      let _ = log sess "sniffing PE file" in
        (* PE header offset is at 0x3c in the MS-DOS compatibility header. *)
      let _ = ar.asm_seek 0x3c in
      let pe_hdr_off = ar.asm_get_u32() in
      let _ = log sess "PE header offset: 0x%x" pe_hdr_off in

      let _ = ar.asm_seek pe_hdr_off in
      let pe_signature = ar.asm_get_zstr_padded 4 in
      let _ = log sess "    PE signature: '%s'" pe_signature in
        if pe_signature = pe_magic
        then (ar.asm_seek 0; Some ar)
        else None
    else
      None
  with
      _ -> None
;;


let get_sections
    (sess:Session.sess)
    (ar:asm_reader)
    : (string,(int*int)) Hashtbl.t =
  let _ = log sess "reading sections" in
  (* PE header offset is at 0x3c in the MS-DOS compatibility header. *)
  let _ = ar.asm_seek 0x3c in
  let pe_hdr_off = ar.asm_get_u32() in
  let _ = log sess "PE header offset: 0x%x" pe_hdr_off in

  let _ = ar.asm_seek pe_hdr_off in
  let pe_signature = ar.asm_get_zstr_padded 4 in
  let _ = log sess "    PE signature: '%s'" pe_signature in
  let _ = assert (pe_signature = pe_magic) in
  let _ = ar.asm_adv_u16() in (* machine type *)

  let num_sections = ar.asm_get_u16() in
  let _ = log sess "    num sections: %d" num_sections in

  let _ = ar.asm_adv_u32() in (* timestamp *)

  let symtab_off = ar.asm_get_u32() in
  let _ = log sess "   symtab offset: 0x%x" symtab_off in

  let num_symbols = ar.asm_get_u32() in
  let _ = log sess "     num symbols: %d" num_symbols in

  let loader_hdr_size = ar.asm_get_u16() in
  let _ = log sess "loader header sz: %d" loader_hdr_size in

  let _ = ar.asm_adv_u16() in (* flags *)
  let sections_off = (ar.asm_get_off()) + loader_hdr_size in

  let sects = Hashtbl.create 0 in

  let _ =
    ar.asm_seek sections_off;
    for i = 0 to (num_sections - 1) do
      (* 
       * Section-name encoding is crazy. ASCII-encoding offsets of
       * long names. See pe_section_header for details.
       *)
      let sect_name =
        let sect_name = ar.asm_get_zstr_padded 8 in
          assert ((String.length sect_name) > 0);
          if sect_name.[0] = '/'
          then
            let off_str =
              String.sub sect_name 1 ((String.length sect_name) - 1)
            in
            let i = int_of_string off_str in
            let curr = ar.asm_get_off() in
              ar.asm_seek (symtab_off + i);
              let ext_name = ar.asm_get_zstr() in
                ar.asm_seek curr;
                ext_name
          else
            sect_name
      in
      let _ = ar.asm_adv_u32() in (* virtual size *)
      let _ = ar.asm_adv_u32() in (* virtual address *)
      let file_sz = ar.asm_get_u32() in
      let file_off = ar.asm_get_u32() in
      let _ = ar.asm_adv_u32() in (* reserved *)
      let _ = ar.asm_adv_u32() in (* reserved *)
      let _ = ar.asm_adv_u32() in (* reserved *)
      let _ = ar.asm_adv_u32() in (* flags *)
        Hashtbl.add sects sect_name (file_off, file_sz);
        log sess "       section %d: %s, size %d, offset 0x%x"
          i sect_name file_sz file_off;
    done
  in
    sects
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
