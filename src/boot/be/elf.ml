(*
 * Module for writing System V ELF files.
 *
 * FIXME: Presently heavily infected with x86 and elf32 specificities,
 * though they are reasonably well marked. Needs to be refactored to
 * depend on abi fields if it's to be usable for other elf
 * configurations.
 *)

open Asm;;
open Common;;

let log (sess:Session.sess) =
  Session.log "obj (elf)"
    sess.Session.sess_log_obj
    sess.Session.sess_log_out
;;

let iflog (sess:Session.sess) (thunk:(unit -> unit)) : unit =
  if sess.Session.sess_log_obj
  then thunk ()
  else ()
;;


(* Fixed sizes of structs involved in elf32 spec. *)
let elf32_ehsize = 52L;;
let elf32_phentsize = 32L;;
let elf32_shentsize = 40L;;
let elf32_symsize = 16L;;
let elf32_rela_entsz = 0xcL;;

type ei_class =
    ELFCLASSNONE
  | ELFCLASS32
  | ELFCLASS64
;;


type ei_data =
    ELFDATANONE
  | ELFDATA2LSB
  | ELFDATA2MSB
;;


let elf_identification sess ei_class ei_data =
  SEQ
    [|
      STRING "\x7fELF";
      BYTES
        [|
          (match ei_class with  (* EI_CLASS *)
               ELFCLASSNONE -> 0
             | ELFCLASS32 -> 1
             | ELFCLASS64 -> 2);
          (match ei_data with   (* EI_DATA *)
               ELFDATANONE -> 0
             | ELFDATA2LSB -> 1
             | ELFDATA2MSB -> 2);

          1;                    (* EI_VERSION = EV_CURRENT *)

                                (* EI_OSABI *)
          (match sess.Session.sess_targ with
               FreeBSD_x86_elf -> 9
             | _ -> 0);

          0;                    (* EI_ABIVERSION *)

          0;                    (* EI_PAD #9 *)
          0;                    (* EI_PAD #A *)
          0;                    (* EI_PAD #B *)
          0;                    (* EI_PAD #C *)
          0;                    (* EI_PAD #D *)
          0;                    (* EI_PAD #E *)
          0;                    (* EI_PAD #F *)
        |]
    |]
;;


type e_type =
    ET_NONE
  | ET_REL
  | ET_EXEC
  | ET_DYN
  | ET_CORE
;;


type e_machine =
    (* Maybe support more later. *)
    EM_NONE
  | EM_386
  | EM_X86_64
;;


type e_version =
    EV_NONE
  | EV_CURRENT
;;


let elf32_header
    ~(sess:Session.sess)
    ~(ei_data:ei_data)
    ~(e_type:e_type)
    ~(e_machine:e_machine)
    ~(e_version:e_version)
    ~(e_entry_fixup:fixup)
    ~(e_phoff_fixup:fixup)
    ~(e_shoff_fixup:fixup)
    ~(e_phnum:int64)
    ~(e_shnum:int64)
    ~(e_shstrndx:int64)
    : frag =
  let elf_header_fixup = new_fixup "elf header" in
  let entry_pos =
    if sess.Session.sess_library_mode
    then (IMM 0L)
    else (M_POS e_entry_fixup)
  in
    DEF
      (elf_header_fixup,
       SEQ [| elf_identification sess ELFCLASS32 ei_data;
              WORD (TY_u16, (IMM (match e_type with
                                      ET_NONE -> 0L
                                    | ET_REL -> 1L
                                    | ET_EXEC -> 2L
                                    | ET_DYN -> 3L
                                    | ET_CORE -> 4L)));
              WORD (TY_u16, (IMM (match e_machine with
                                      EM_NONE -> 0L
                                    | EM_386 -> 3L
                                    | EM_X86_64 -> 62L)));
              WORD (TY_u32, (IMM (match e_version with
                                      EV_NONE -> 0L
                                    | EV_CURRENT -> 1L)));
              WORD (TY_u32, entry_pos);
              WORD (TY_u32, (F_POS e_phoff_fixup));
              WORD (TY_u32, (F_POS e_shoff_fixup));
              WORD (TY_u32, (IMM 0L)); (* e_flags *)
              WORD (TY_u16, (IMM elf32_ehsize));
              WORD (TY_u16, (IMM elf32_phentsize));
              WORD (TY_u16, (IMM e_phnum));
              WORD (TY_u16, (IMM elf32_shentsize));
              WORD (TY_u16, (IMM e_shnum));
              WORD (TY_u16, (IMM e_shstrndx));
           |])
;;


type sh_type =
    SHT_NULL
  | SHT_PROGBITS
  | SHT_SYMTAB
  | SHT_STRTAB
  | SHT_RELA
  | SHT_HASH
  | SHT_DYNAMIC
  | SHT_NOTE
  | SHT_NOBITS
  | SHT_REL
  | SHT_SHLIB
  | SHT_DYNSYM
;;


type sh_flags =
    SHF_WRITE
  | SHF_ALLOC
  | SHF_EXECINSTR
;;


let section_header
    ?(sh_link:int64 option=None)
    ?(sh_info:int64 option=None)
    ?(zero_sh_addr:bool=false)
    ?(sh_flags:sh_flags list=[])
    ?(section_fixup:fixup option=None)
    ?(sh_addralign:int64=1L)
    ?(sh_entsize:int64=0L)
    ~(shstring_table_fixup:fixup)
    ~(shname_string_fixup:fixup)
    (sh_type:sh_type)
    : frag =
  SEQ
    [|
      WORD (TY_i32, (SUB
                       ((F_POS shname_string_fixup),
                        (F_POS shstring_table_fixup))));
      WORD (TY_u32, (IMM (match sh_type with
                              SHT_NULL -> 0L
                            | SHT_PROGBITS -> 1L
                            | SHT_SYMTAB -> 2L
                            | SHT_STRTAB -> 3L
                            | SHT_RELA -> 4L
                            | SHT_HASH -> 5L
                            | SHT_DYNAMIC -> 6L
                            | SHT_NOTE -> 7L
                            | SHT_NOBITS -> 8L
                            | SHT_REL -> 9L
                            | SHT_SHLIB -> 10L
                            | SHT_DYNSYM -> 11L)));
      WORD (TY_u32, (IMM (fold_flags
                            (fun f -> match f with
                                 SHF_WRITE -> 0x1L
                               | SHF_ALLOC -> 0x2L
                               | SHF_EXECINSTR -> 0x4L) sh_flags)));
      WORD (TY_u32,
            if zero_sh_addr
            then IMM 0L
            else (match section_fixup with
                      None -> (IMM 0L)
                    | Some s -> (M_POS s)));
      WORD (TY_u32, (match section_fixup with
                         None -> (IMM 0L)
                       | Some s -> (F_POS s)));
      WORD (TY_u32, (match section_fixup with
                         None -> (IMM 0L)
                       | Some s -> (F_SZ s)));
      WORD (TY_u32, (IMM (match sh_link with
                              None -> 0L
                            | Some i -> i)));
      WORD (TY_u32, (IMM (match sh_info with
                              None -> 0L
                            | Some i -> i)));
      WORD (TY_u32, (IMM sh_addralign));
      WORD (TY_u32, (IMM sh_entsize));
    |]
;;


type p_type =
    PT_NULL
  | PT_LOAD
  | PT_DYNAMIC
  | PT_INTERP
  | PT_NOTE
  | PT_SHLIB
  | PT_PHDR
;;


type p_flag =
    PF_X
  | PF_W
  | PF_R
;;


let program_header
    ~(p_type:p_type)
    ~(segment_fixup:fixup)
    ~(p_flags:p_flag list)
    ~(p_align:int64)
    : frag =
  SEQ
    [|
      WORD (TY_u32, (IMM (match p_type with
                              PT_NULL -> 0L
                            | PT_LOAD -> 1L
                            | PT_DYNAMIC -> 2L
                            | PT_INTERP -> 3L
                            | PT_NOTE -> 4L
                            | PT_SHLIB -> 5L
                            | PT_PHDR -> 6L)));
      WORD (TY_u32, (F_POS segment_fixup));
      WORD (TY_u32, (M_POS segment_fixup));
      WORD (TY_u32, (M_POS segment_fixup));
      WORD (TY_u32, (F_SZ segment_fixup));
      WORD (TY_u32, (M_SZ segment_fixup));
      WORD (TY_u32, (IMM (fold_flags
                            (fun f ->
                               match f with
                                   PF_X -> 0x1L
                                 | PF_W -> 0x2L
                                 | PF_R -> 0x4L)
                            p_flags)));
      WORD (TY_u32, (IMM p_align));
    |]
;;


type st_bind =
    STB_LOCAL
  | STB_GLOBAL
  | STB_WEAK
;;


type st_type =
    STT_NOTYPE
  | STT_OBJECT
  | STT_FUNC
  | STT_SECTION
  | STT_FILE
;;


(* Special symbol-section indices *)
let shn_UNDEF   = 0L;;
let shn_ABS     = 0xfff1L;;
let shn_ABS     = 0xfff2L;;


let symbol
    ~(string_table_fixup:fixup)
    ~(name_string_fixup:fixup)
    ~(sym_target_fixup:fixup option)
    ~(st_bind:st_bind)
    ~(st_type:st_type)
    ~(st_shndx:int64)
    : frag =
  let st_bind_num =
    match st_bind with
        STB_LOCAL -> 0L
      | STB_GLOBAL -> 1L
      | STB_WEAK -> 2L
  in
  let st_type_num =
    match st_type with
        STT_NOTYPE -> 0L
      | STT_OBJECT -> 1L
      | STT_FUNC -> 2L
      | STT_SECTION -> 3L
      | STT_FILE -> 4L
  in
    SEQ
      [|
        WORD (TY_u32, (SUB
                         ((F_POS name_string_fixup),
                          (F_POS string_table_fixup))));
        WORD (TY_u32, (match sym_target_fixup with
                           None -> (IMM 0L)
                         | Some f -> (M_POS f)));
        WORD (TY_u32, (match sym_target_fixup with
                           None -> (IMM 0L)
                         | Some f -> (M_SZ f)));
        WORD (TY_u8,           (* st_info *)
              (OR
                 ((SLL ((IMM st_bind_num), 4)),
                  (AND ((IMM st_type_num), (IMM 0xfL))))));
        WORD (TY_u8, (IMM 0L)); (* st_other *)
        WORD (TY_u16, (IMM st_shndx));
      |]
;;

type d_tag =
    DT_NULL
  | DT_NEEDED
  | DT_PLTRELSZ
  | DT_PLTGOT
  | DT_HASH
  | DT_STRTAB
  | DT_SYMTAB
  | DT_RELA
  | DT_RELASZ
  | DT_RELAENT
  | DT_STRSZ
  | DT_SYMENT
  | DT_INIT
  | DT_FINI
  | DT_SONAME
  | DT_RPATH
  | DT_SYMBOLIC
  | DT_REL
  | DT_RELSZ
  | DT_RELENT
  | DT_PLTREL
  | DT_DEBUG
  | DT_TEXTREL
  | DT_JMPREL
  | DT_BIND_NOW
  | DT_INIT_ARRAY
  | DT_FINI_ARRAY
  | DT_INIT_ARRAYSZ
  | DT_FINI_ARRAYSZ
  | DT_RUNPATH
  | DT_FLAGS
  | DT_ENCODING
  | DT_PREINIT_ARRAY
  | DT_PREINIT_ARRAYSZ
;;

type elf32_dyn = (d_tag * expr64);;

let elf32_num_of_dyn_tag tag =
  match tag with
      DT_NULL -> 0L
    | DT_NEEDED -> 1L
    | DT_PLTRELSZ -> 2L
    | DT_PLTGOT -> 3L
    | DT_HASH -> 4L
    | DT_STRTAB -> 5L
    | DT_SYMTAB -> 6L
    | DT_RELA -> 7L
    | DT_RELASZ -> 8L
    | DT_RELAENT -> 9L
    | DT_STRSZ -> 10L
    | DT_SYMENT -> 11L
    | DT_INIT -> 12L
    | DT_FINI -> 13L
    | DT_SONAME -> 14L
    | DT_RPATH -> 15L
    | DT_SYMBOLIC -> 16L
    | DT_REL -> 17L
    | DT_RELSZ -> 18L
    | DT_RELENT -> 19L
    | DT_PLTREL -> 20L
    | DT_DEBUG -> 21L
    | DT_TEXTREL -> 22L
    | DT_JMPREL -> 23L
    | DT_BIND_NOW -> 24L
    | DT_INIT_ARRAY -> 25L
    | DT_FINI_ARRAY -> 26L
    | DT_INIT_ARRAYSZ -> 27L
    | DT_FINI_ARRAYSZ -> 28L
    | DT_RUNPATH -> 29L
    | DT_FLAGS -> 30L
    | DT_ENCODING -> 31L
    | DT_PREINIT_ARRAY -> 32L
    | DT_PREINIT_ARRAYSZ -> 33L
;;

let elf32_dyn_frag d =
  let (tag, expr) = d in
  let tagval = elf32_num_of_dyn_tag tag in
    SEQ [| WORD (TY_u32, (IMM tagval)); WORD (TY_u32, expr) |]
;;

type elf32_386_reloc_type =
    R_386_NONE
  | R_386_32
  | R_386_PC32
  | R_386_GOT32
  | R_386_PLT32
  | R_386_COPY
  | R_386_GLOB_DAT
  | R_386_JMP_SLOT
  | R_386_RELATIVE
  | R_386_GOTOFF
  | R_386_GOTPC
;;


type elf32_386_rela =
    { elf32_386_rela_type: elf32_386_reloc_type;
      elf32_386_rela_offset: expr64;
      elf32_386_rela_sym: expr64;
      elf32_386_rela_addend: expr64 }
;;

let elf32_386_rela_frag r =
  let type_val =
    match r.elf32_386_rela_type with
        R_386_NONE -> 0L
      | R_386_32 -> 1L
      | R_386_PC32 -> 2L
      | R_386_GOT32 -> 3L
      | R_386_PLT32 -> 4L
      | R_386_COPY -> 5L
      | R_386_GLOB_DAT -> 6L
      | R_386_JMP_SLOT -> 7L
      | R_386_RELATIVE -> 8L
      | R_386_GOTOFF -> 9L
      | R_386_GOTPC -> 10L
  in
  let info_expr =
    WORD (TY_u32,
          (OR
             (SLL ((r.elf32_386_rela_sym), 8),
              AND ((IMM 0xffL), (IMM type_val)))))
  in
    SEQ [| WORD (TY_u32, r.elf32_386_rela_offset);
           info_expr;
           WORD (TY_u32, r.elf32_386_rela_addend) |]
;;


let elf32_linux_x86_file
    ~(sess:Session.sess)
    ~(crate:Ast.crate)
    ~(entry_name:string)
    ~(text_frags:(string option, frag) Hashtbl.t)
    ~(data_frags:(string option, frag) Hashtbl.t)
    ~(bss_frags:(string option, frag) Hashtbl.t)
    ~(rodata_frags:(string option, frag) Hashtbl.t)
    ~(required_fixups:(string, fixup) Hashtbl.t)
    ~(dwarf:Dwarf.debug_records)
    ~(sem:Semant.ctxt)
    ~(needed_libs:string array)
    : frag =

  (* Procedure Linkage Tables (PLTs), Global Offset Tables
   * (GOTs), and the relocations that set them up:
   *
   * The PLT goes in a section called .plt and GOT in a section called
   * .got. The portion of the GOT that holds PLT jump slots goes in a
   * section called .got.plt. Dynamic relocations for these jump slots go in
   * section .rela.plt.
   *
   * The easiest way to understand the PLT/GOT system is to draw it:
   *
   *     PLT                          GOT
   *   +----------------------+     +----------------------+
   *  0| push &<GOT[1]>            0| <reserved>
   *   | jmp *GOT[2]               1| <libcookie>
   *   |                           2| & <ld.so:resolve-a-sym>
   *  1| jmp *GOT[3]               3| & <'push 0' in PLT[1]>
   *   | push 0                    4| & <'push 1' in PLT[2]>
   *   | jmp *PLT[0]               5| & <'push 2' in PLT[3]>
   *   |
   *  2| jmp *GOT[4]
   *   | push 1
   *   | jmp *PLT[0]
   *   |
   *  2| jmp *GOT[5]
   *   | push 2
   *   | jmp *PLT[0]
   *
   *
   * In normal user code, we call PLT entries with a call to a
   * PC-relative address, the PLT entry, which itself does an indirect
   * jump through a slot in the GOT that it also addresses
   * PC-relative. This makes the whole scheme PIC.
   *
   * The linker fills in the GOT on startup. For the first 3, it uses
   * its own thinking. For the remainder it needs to be instructed to
   * fill them in with "jump slot relocs", type R_386_JUMP_SLOT, each
   * of which says in effect which PLT entry it's to point back to and
   * which symbol it's to be resolved to later. These relocs go in the
   * section .rela.plt.
   *)

    let plt0_fixup = new_fixup "PLT[0]" in
    let got_prefix = SEQ [| WORD (TY_u32, (IMM 0L));
                            WORD (TY_u32, (IMM 0L));
                            WORD (TY_u32, (IMM 0L)); |]
    in

    let got_cell reg i =
      let got_entry_off = Int64.of_int (i*4) in
      let got_entry_mem = Il.RegIn (reg, (Some (Asm.IMM got_entry_off))) in
        Il.Mem (got_entry_mem, Il.ScalarTy (Il.AddrTy Il.CodeTy))
    in

    let got_code_cell reg i =
      Il.CodePtr (Il.Cell (got_cell reg i))
    in

    let plt0_frag =
      let reg = Il.Hreg X86.eax in
      let e = X86.new_emitter_without_vregs () in
        Il.emit e (Il.Push (Il.Cell (got_cell reg 1)));
        Il.emit e (Il.jmp Il.JMP (got_code_cell reg 2));
        Il.emit e Il.Nop;
        Il.emit e Il.Nop;
        Il.emit e Il.Nop;
        Il.emit e Il.Nop;
        DEF (plt0_fixup, (X86.frags_of_emitted_quads sess e))
    in

  (*
   * The existence of the GOT/PLT mish-mash causes, therefore, the
   * following new sections:
   *
   *   .plt       - the PLT itself, in the r/x text segment
   *   .got.plt   - the PLT-used portion of the GOT, in the r/w segment
   *   .rela.plt  - the dynamic relocs for the GOT-PLT, in the r/x segment
   *
   * In addition, because we're starting up a dynamically linked executable,
   * we have to have several more sections!
   *
   *   .interp    - the read-only section that names ld.so
   *   .dynsym    - symbols named by the PLT/GOT entries, r/x segment
   *   .dynstr    - string-names used in those symbols, r/x segment
   *   .hash      - hashtable in which to look these up, r/x segment
   *   .dynamic   - the machine-readable description of the dynamic
   *                linkage requirements of this elf file, in the
   *                r/w _DYNAMIC segment
   *
   * The Dynamic section contains a sequence of 2-word records of type
   * d_tag.
   *
   *)

    (* There are 17 official section headers in the file we're making:  *)
    (*                                                                  *)
    (* section 0: <null section>                                        *)
    (*                                                                  *)
    (* section 1:  .interp            (segment 1: R+X, INTERP)          *)
    (*                                                                  *)
    (* section 2:  .text              (segment 2: R+X, LOAD)            *)
    (* section 3:  .rodata                   ...                        *)
    (* section 4:  .dynsym                   ...                        *)
    (* section 5:  .dynstr                   ...                        *)
    (* section 6:  .hash                     ...                        *)
    (* section 7:  .plt                      ...                        *)
    (* section 8:  .got                      ...                        *)
    (* section 9:  .rela.plt                 ...                        *)
    (*                                                                  *)
    (* section 10: .data              (segment 3: R+W, LOAD)            *)
    (* section 11: .bss                      ...                        *)
    (*                                                                  *)
    (* section 12: .dynamic           (segment 4: R+W, DYNAMIC)         *)
    (*                                                                  *)
    (* section 13: .shstrtab          (not in a segment)                *)
    (* section 14: .debug_aranges     (segment 2: cont'd)               *)
    (* section 15: .debug_pubnames           ...                        *)
    (* section 14: .debug_info               ...                        *)
    (* section 15: .debug_abbrev             ...                        *)
    (* section 14: .debug_line               ...                        *)
    (* section 15: .debug_frame              ...                        *)
    (* section 16: .note..rust        (segment 5: NOTE)                 *)

    let sname s =
      new_fixup (Printf.sprintf "string name of '%s' section" s)
    in
    let null_section_name_fixup = sname "<null>" in
    let interp_section_name_fixup = sname ".interp"in
    let text_section_name_fixup = sname ".text" in
    let rodata_section_name_fixup = sname ".rodata" in
    let dynsym_section_name_fixup = sname ".dynsym" in
    let dynstr_section_name_fixup = sname ".dynstr" in
    let hash_section_name_fixup = sname ".hash" in
    let plt_section_name_fixup = sname ".plt" in
    let got_plt_section_name_fixup = sname ".got.plt" in
    let rela_plt_section_name_fixup = sname ".rela.plt" in
    let data_section_name_fixup = sname ".data" in
    let bss_section_name_fixup = sname ".bss" in
    let dynamic_section_name_fixup = sname ".dynamic" in
    let shstrtab_section_name_fixup = sname ".shstrtab" in
    let debug_aranges_section_name_fixup = sname ".debug_aranges" in
    let debug_pubnames_section_name_fixup = sname ".debug_pubnames" in
    let debug_info_section_name_fixup = sname ".debug_info" in
    let debug_abbrev_section_name_fixup = sname ".debug_abbrev" in
    let debug_line_section_name_fixup = sname ".debug_line" in
    let debug_frame_section_name_fixup = sname ".debug_frame" in
    let note_rust_section_name_fixup = sname ".note.rust" in

  (* let interpndx      = 1L in *)  (* Section index of .interp *)
  let textndx        = 2L in  (* Section index of .text *)
  let rodatandx      = 3L in  (* Section index of .rodata *)
  let dynsymndx      = 4L in  (* Section index of .dynsym *)
  let dynstrndx      = 5L in  (* Section index of .dynstr *)
  (* let hashndx        = 6L in *)  (* Section index of .hash *)
  let pltndx         = 7L in  (* Section index of .plt *)
  (* let gotpltndx      = 8L in *)  (* Section index of .got.plt *)
  (* let relapltndx     = 9L in *)  (* Section index of .rela.plt *)
  let datandx        = 10L in  (* Section index of .data *)
  let bssndx         = 11L in  (* Section index of .bss *)
  (* let dynamicndx     = 12L in *) (* Section index of .dynamic *)
  let shstrtabndx    = 13L in (* Section index of .shstrtab *)

  let section_header_table_fixup = new_fixup ".section header table" in
  let interp_section_fixup = new_fixup ".interp section" in
  let text_section_fixup = new_fixup ".text section" in
  let rodata_section_fixup = new_fixup ".rodata section" in
  let dynsym_section_fixup = new_fixup ".dynsym section" in
  let dynstr_section_fixup = new_fixup ".dynstr section" in
  let hash_section_fixup = new_fixup ".hash section" in
  let plt_section_fixup = new_fixup ".plt section" in
  let got_plt_section_fixup = new_fixup ".got.plt section" in
  let rela_plt_section_fixup = new_fixup ".rela.plt section" in
  let data_section_fixup = new_fixup ".data section" in
  let bss_section_fixup = new_fixup ".bss section" in
  let dynamic_section_fixup = new_fixup ".dynamic section" in
  let shstrtab_section_fixup = new_fixup ".shstrtab section" in
  let note_rust_section_fixup = new_fixup ".shstrtab section" in

  let shstrtab_section =
    SEQ
      [|
        DEF (null_section_name_fixup, ZSTRING "");
        DEF (interp_section_name_fixup, ZSTRING ".interp");
        DEF (text_section_name_fixup, ZSTRING ".text");
        DEF (rodata_section_name_fixup, ZSTRING ".rodata");
        DEF (dynsym_section_name_fixup, ZSTRING ".dynsym");
        DEF (dynstr_section_name_fixup, ZSTRING ".dynstr");
        DEF (hash_section_name_fixup, ZSTRING ".hash");
        DEF (plt_section_name_fixup, ZSTRING ".plt");
        DEF (got_plt_section_name_fixup, ZSTRING ".got.plt");
        DEF (rela_plt_section_name_fixup, ZSTRING ".rela.plt");
        DEF (data_section_name_fixup, ZSTRING ".data");
        DEF (bss_section_name_fixup, ZSTRING ".bss");
        DEF (dynamic_section_name_fixup, ZSTRING ".dynamic");
        DEF (shstrtab_section_name_fixup, ZSTRING ".shstrtab");
        DEF (debug_aranges_section_name_fixup, ZSTRING ".debug_aranges");
        DEF (debug_pubnames_section_name_fixup, ZSTRING ".debug_pubnames");
        DEF (debug_info_section_name_fixup, ZSTRING ".debug_info");
        DEF (debug_abbrev_section_name_fixup, ZSTRING ".debug_abbrev");
        DEF (debug_line_section_name_fixup, ZSTRING ".debug_line");
        DEF (debug_frame_section_name_fixup, ZSTRING ".debug_frame");
        DEF (note_rust_section_name_fixup, ZSTRING ".note.rust");
      |]
  in

  let section_headers =
    [|
        (* <null> *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: null_section_name_fixup
           ~section_fixup: None
           ~sh_addralign: 0L
           SHT_NULL);

        (* .interp *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: interp_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some interp_section_fixup)
           SHT_PROGBITS);

        (* .text *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: text_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_EXECINSTR ]
           ~section_fixup: (Some text_section_fixup)
           ~sh_addralign: 32L
           SHT_PROGBITS);

        (* .rodata *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: rodata_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some rodata_section_fixup)
           ~sh_addralign: 32L
           SHT_PROGBITS);

        (* .dynsym *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: dynsym_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some dynsym_section_fixup)
           ~sh_addralign: 4L
           ~sh_entsize: elf32_symsize
           ~sh_link: (Some dynstrndx)
           SHT_DYNSYM);

        (* .dynstr *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: dynstr_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some dynstr_section_fixup)
           SHT_STRTAB);

        (* .hash *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: hash_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some hash_section_fixup)
           ~sh_addralign: 4L
           ~sh_entsize: 4L
           SHT_PROGBITS);

        (* .plt *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: plt_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_EXECINSTR ]
           ~section_fixup: (Some plt_section_fixup)
           ~sh_addralign: 4L
           SHT_PROGBITS);

        (* .got.plt *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: got_plt_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_WRITE ]
           ~section_fixup: (Some got_plt_section_fixup)
           ~sh_addralign: 4L
           SHT_PROGBITS);

        (* .rela.plt *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: rela_plt_section_name_fixup
           ~sh_flags: [ SHF_ALLOC ]
           ~section_fixup: (Some rela_plt_section_fixup)
           ~sh_addralign: 4L
           ~sh_entsize: elf32_rela_entsz
           ~sh_link: (Some dynsymndx)
           ~sh_info: (Some pltndx)
           SHT_RELA);

        (* .data *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: data_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_WRITE ]
           ~section_fixup: (Some data_section_fixup)
           ~sh_addralign: 32L
           SHT_PROGBITS);

        (* .bss *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: bss_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_WRITE ]
           ~section_fixup: (Some bss_section_fixup)
           ~sh_addralign: 32L
           SHT_NOBITS);

        (* .dynamic *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: dynamic_section_name_fixup
           ~sh_flags: [ SHF_ALLOC; SHF_WRITE ]
           ~section_fixup: (Some dynamic_section_fixup)
           ~sh_addralign: 8L
           ~sh_link: (Some dynstrndx)
           SHT_DYNAMIC);

        (* .shstrtab *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: shstrtab_section_name_fixup
           ~section_fixup: (Some shstrtab_section_fixup)
           SHT_STRTAB);

(* 
   FIXME: uncomment the dwarf section headers as you make use of them;
   recent gdb versions have got fussier about parsing dwarf and don't
   like seeing junk there. 
*)

        (* .debug_aranges *)
(*

        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_aranges_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_aranges_fixup)
           ~sh_addralign: 8L
           ~zero_sh_addr: true
           SHT_PROGBITS);
*)
        (* .debug_pubnames *)
(*
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_pubnames_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_pubnames_fixup)
           ~zero_sh_addr: true
           SHT_PROGBITS);
*)

        (* .debug_info *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_info_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_info_fixup)
           ~zero_sh_addr: true
           SHT_PROGBITS);

        (* .debug_abbrev *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_abbrev_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_abbrev_fixup)
           ~zero_sh_addr: true
           SHT_PROGBITS);

        (* .debug_line *)
(*
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_line_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_line_fixup)
           ~zero_sh_addr: true
           SHT_PROGBITS);
*)

        (* .debug_frame *)
(*
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: debug_frame_section_name_fixup
           ~section_fixup: (Some sem.Semant.ctxt_debug_frame_fixup)
           ~sh_addralign: 4L
           ~zero_sh_addr: true
           SHT_PROGBITS);
*)

        (* .note.rust *)
        (section_header
           ~shstring_table_fixup: shstrtab_section_fixup
           ~shname_string_fixup: note_rust_section_name_fixup
           ~section_fixup: (Some note_rust_section_fixup)
           SHT_NOTE);

      |]
  in
  let section_header_table = SEQ section_headers in


  (* There are 6 official program headers in the file we're making:   *)
  (* segment 0: RX / PHDR                                             *)
  (* segment 1: R  / INTERP                                           *)
  (* segment 2: RX / LOAD                                             *)
  (* segment 3: RW / LOAD                                             *)
  (* segment 4: RW / DYNAMIC                                          *)
  (* segment 5: R                                                     *)

  let program_header_table_fixup = new_fixup "program header table" in
  let segment_0_fixup = new_fixup "segment 0" in
  let segment_1_fixup = new_fixup "segment 1" in
  let segment_2_fixup = new_fixup "segment 2" in
  let segment_3_fixup = new_fixup "segment 3" in
  let segment_4_fixup = new_fixup "segment 4" in
  let segment_5_fixup = new_fixup "segment 5" in

  let segment_0_align = 4 in
  let segment_1_align = 1 in
  let segment_2_align = 0x1000 in
  let segment_3_align = 0x1000 in
  let segment_4_align = 0x1000 in
  let segment_5_align = 1 in

  let program_headers = [|
        (program_header
           ~p_type: PT_PHDR
           ~segment_fixup: segment_0_fixup
           ~p_flags: [ PF_R; PF_X ]
           ~p_align: (Int64.of_int segment_0_align));
        (program_header
           ~p_type: PT_INTERP
           ~segment_fixup: segment_1_fixup
           ~p_flags: [ PF_R ]
           ~p_align: (Int64.of_int segment_1_align));
        (program_header
           ~p_type: PT_LOAD
           ~segment_fixup: segment_2_fixup
           ~p_flags: [ PF_R; PF_X ]
           ~p_align: (Int64.of_int segment_2_align));
        (program_header
           ~p_type: PT_LOAD
           ~segment_fixup: segment_3_fixup
           ~p_flags: [ PF_R; PF_W ]
           ~p_align: (Int64.of_int segment_3_align));
        (program_header
           ~p_type: PT_DYNAMIC
           ~segment_fixup: segment_4_fixup
           ~p_flags: [ PF_R; PF_W ]
           ~p_align: (Int64.of_int segment_4_align));
        (program_header
           ~p_type: PT_NOTE
           ~segment_fixup: segment_5_fixup
           ~p_flags: [ PF_R;]
           ~p_align: (Int64.of_int segment_5_align));
      |]
  in
  let program_header_table = SEQ program_headers in

  let e_entry_fixup = new_fixup "entry symbol" in

  let elf_header =
    elf32_header
      ~sess
      ~ei_data: ELFDATA2LSB
      ~e_type: (if sess.Session.sess_library_mode then ET_DYN else ET_EXEC)
      ~e_machine: EM_386
      ~e_version: EV_CURRENT

      ~e_entry_fixup: e_entry_fixup
      ~e_phoff_fixup: program_header_table_fixup
      ~e_shoff_fixup: section_header_table_fixup
      ~e_phnum: (Int64.of_int (Array.length program_headers))
      ~e_shnum: (Int64.of_int (Array.length section_headers))
      ~e_shstrndx: shstrtabndx
  in

  let n_syms = ref 1 in (* The empty symbol, implicit. *)

  let data_sym name st_bind fixup =
    let name_fixup = new_fixup ("data symbol name fixup: '" ^ name ^ "'") in
    let strtab_entry = DEF (name_fixup, ZSTRING name) in
    let symtab_entry =
      symbol
        ~string_table_fixup: dynstr_section_fixup
        ~name_string_fixup: name_fixup
        ~sym_target_fixup: (Some fixup)
        ~st_bind
        ~st_type: STT_OBJECT
        ~st_shndx: datandx
    in
      incr n_syms;
      (strtab_entry, symtab_entry)
  in

  let bss_sym name st_bind fixup =
    let name_fixup = new_fixup ("bss symbol name fixup: '" ^ name ^ "'") in
    let strtab_entry = DEF (name_fixup, ZSTRING name) in
    let symtab_entry =
      symbol
        ~string_table_fixup: dynstr_section_fixup
        ~name_string_fixup: name_fixup
        ~sym_target_fixup: (Some fixup)
        ~st_bind
        ~st_type: STT_OBJECT
        ~st_shndx: bssndx
    in
      incr n_syms;
      (strtab_entry, symtab_entry)
  in

  let rodata_sym name st_bind fixup =
    let name_fixup = new_fixup ("rodata symbol name fixup: '" ^ name ^ "'") in
    let strtab_entry = DEF (name_fixup, ZSTRING name) in
    let symtab_entry =
      symbol
        ~string_table_fixup: dynstr_section_fixup
        ~name_string_fixup: name_fixup
        ~sym_target_fixup: (Some fixup)
        ~st_bind
        ~st_type: STT_OBJECT
        ~st_shndx: rodatandx
    in
      incr n_syms;
      (strtab_entry, symtab_entry)
  in

  let text_sym name st_bind fixup =
    let name_fixup = new_fixup ("text symbol name fixup: '" ^ name ^ "'") in
    let strtab_frag = DEF (name_fixup, ZSTRING name) in
    let symtab_frag =
      symbol
        ~string_table_fixup: dynstr_section_fixup
        ~name_string_fixup: name_fixup
        ~sym_target_fixup: (Some fixup)
        ~st_bind: st_bind
        ~st_type: STT_FUNC
        ~st_shndx: textndx
    in
      incr n_syms;
      (strtab_frag, symtab_frag)
  in

  let require_sym name st_bind _(*fixup*) =
    let name_fixup =
      new_fixup ("require symbol name fixup: '" ^ name ^ "'")
    in
    let strtab_frag = DEF (name_fixup, ZSTRING name) in
    let symtab_frag =
      symbol
        ~string_table_fixup: dynstr_section_fixup
        ~name_string_fixup: name_fixup
        ~sym_target_fixup: None
        ~st_bind
        ~st_type: STT_FUNC
        ~st_shndx: shn_UNDEF
    in
      incr n_syms;
      (strtab_frag, symtab_frag)
  in

  let frags_of_symbol sym_emitter st_bind symname_opt symbody x =
    let (strtab_frags, symtab_frags, body_frags) = x in
    let (strtab_frag, symtab_frag, body_frag) =
      match symname_opt with
          None -> (MARK, MARK, symbody)
        | Some symname ->
            let body_fixup =
              new_fixup ("symbol body fixup: '" ^ symname ^ "'")
            in
            let body =
              if symname = entry_name
              then DEF (e_entry_fixup, DEF (body_fixup, symbody))
              else DEF (body_fixup, symbody)
            in
            let (str, sym) = sym_emitter symname st_bind body_fixup in
              (str, sym, body)
    in
      ((strtab_frag :: strtab_frags),
       (symtab_frag :: symtab_frags),
       (body_frag :: body_frags))
  in

  let frags_of_require_symbol sym_emitter st_bind symname plt_entry_fixup x =
    let (i, strtab_frags, symtab_frags,
         plt_frags, got_plt_frags, rela_plt_frags) = x in
    let (strtab_frag, symtab_frag) = sym_emitter symname st_bind None in
    let e = X86.new_emitter_without_vregs () in
    let jump_slot_fixup = new_fixup ("jump slot #" ^ string_of_int i) in
    let jump_slot_initial_target_fixup =
      new_fixup ("jump slot #" ^ string_of_int i ^ " initial target") in

    (* You may notice this PLT entry doesn't look like either of the
     * types of "normal" PLT entries outlined in the ELF manual. It is,
     * however, just what you get when you combine a PIC PLT entry with
     * inline calls to the horrible __i686.get_pc_thunk.ax kludge used
     * on x86 to support entering PIC PLTs. We're just doing it *in*
     * the PLT entries rather than infecting all the callers with the
     * obligation of having the GOT address in a register on
     * PLT-entry.
     *)

    let plt_frag =
      let (reg, _, _) = X86.get_next_pc_thunk in

        Il.emit_full e (Some plt_entry_fixup) Il.Dead;

        Abi.load_fixup_addr e reg got_plt_section_fixup Il.CodeTy;

        Il.emit e (Il.jmp Il.JMP (got_code_cell reg (2+i)));

        Il.emit_full e (Some jump_slot_initial_target_fixup)
          (Il.Push (X86.immi (Int64.of_int i)));

        Il.emit e (Il.jmp Il.JMP (Il.direct_code_ptr plt0_fixup));
        X86.frags_of_emitted_quads sess e
    in
    let got_plt_frag =
      DEF (jump_slot_fixup,
           WORD (TY_u32, (M_POS jump_slot_initial_target_fixup)))
    in
    let rela_plt =
      { elf32_386_rela_type = R_386_JMP_SLOT;
        elf32_386_rela_offset = (M_POS jump_slot_fixup);
        elf32_386_rela_sym = (IMM (Int64.of_int i));
        elf32_386_rela_addend = (IMM 0L) }
    in
    let rela_plt_frag = elf32_386_rela_frag rela_plt in
      (i+1,
       (strtab_frag :: strtab_frags),
       (symtab_frag :: symtab_frags),
       (plt_frag :: plt_frags),
       (got_plt_frag :: got_plt_frags),
       (rela_plt_frag :: rela_plt_frags))
  in

  (* Emit text export symbols. *)
  let (global_text_strtab_frags, global_text_symtab_frags) =
    match htab_search sem.Semant.ctxt_native_provided SEG_text with
        None -> ([], [])
      | Some etab ->
          Hashtbl.fold
            begin
              fun name fix x ->
                let (strtab_frags, symtab_frags) = x in
                let (str, sym) = text_sym name STB_GLOBAL fix in
                  (str :: strtab_frags,
                   sym :: symtab_frags)
            end
            etab
            ([],[])
  in

  (* Emit text fragments (possibly named). *)
  let (global_text_strtab_frags,
       global_text_symtab_frags,
       text_body_frags) =
    Hashtbl.fold
      (frags_of_symbol text_sym STB_GLOBAL)
      text_frags
      (global_text_strtab_frags, global_text_symtab_frags, [])
  in

  let (local_text_strtab_frags,
       local_text_symtab_frags) =

    let symbol_frags_of_code _ code accum =
      let (strtab_frags, symtab_frags) = accum in
      let fix = code.Semant.code_fixup in
      let (strtab_frag, symtab_frag) =
        text_sym fix.fixup_name STB_LOCAL fix
      in
      (strtab_frag :: strtab_frags,
       symtab_frag :: symtab_frags)
    in

    let symbol_frags_of_glue_code g code accum =
      let (strtab_frags, symtab_frags) = accum in
      let fix = code.Semant.code_fixup in
      let (strtab_frag, symtab_frag) =
        text_sym (Semant.glue_str sem g) STB_LOCAL fix
      in
      (strtab_frag :: strtab_frags,
       symtab_frag :: symtab_frags)
    in

    let item_str_frags, item_sym_frags =
      Hashtbl.fold symbol_frags_of_code
        sem.Semant.ctxt_all_item_code ([], [])
    in
    let glue_str_frags, glue_sym_frags =
      Hashtbl.fold symbol_frags_of_glue_code
        sem.Semant.ctxt_glue_code ([], [])
    in
      (item_str_frags @ glue_str_frags,
       item_sym_frags @ glue_sym_frags)
  in

  (* Emit rodata export symbols. *)
  let (rodata_strtab_frags, rodata_symtab_frags) =
    match htab_search sem.Semant.ctxt_native_provided SEG_data with
        None -> ([], [])
      | Some etab ->
          Hashtbl.fold
            begin
              fun name fix x ->
                let (strtab_frags, symtab_frags) = x in
                let (str, sym) = rodata_sym name STB_GLOBAL fix in
                  (str :: strtab_frags,
                   sym :: symtab_frags)
            end
            etab
            ([],[])
  in

  (* Emit rodata fragments (possibly named). *)
  let (rodata_strtab_frags,
       rodata_symtab_frags,
       rodata_body_frags) =
    Hashtbl.fold
      (frags_of_symbol rodata_sym STB_GLOBAL)
      rodata_frags
      (rodata_strtab_frags, rodata_symtab_frags, [])
  in


  let (data_strtab_frags,
       data_symtab_frags,
       data_body_frags) =
    Hashtbl.fold (frags_of_symbol data_sym STB_GLOBAL) data_frags ([],[],[])
  in

  let (bss_strtab_frags,
       bss_symtab_frags,
       bss_body_frags) =
    Hashtbl.fold (frags_of_symbol bss_sym STB_GLOBAL) bss_frags ([],[],[])
  in

  let (_,
       require_strtab_frags,
       require_symtab_frags,
       plt_frags,
       got_plt_frags,
       rela_plt_frags) =
    Hashtbl.fold (frags_of_require_symbol require_sym STB_GLOBAL)
      required_fixups
      (1,[],[],[plt0_frag],[got_prefix],[])
  in
  let require_symtab_frags = List.rev require_symtab_frags in
  let plt_frags = List.rev plt_frags in
  let got_plt_frags = List.rev got_plt_frags in
  let rela_plt_frags = List.rev rela_plt_frags in

  let dynamic_needed_strtab_frags =
    Array.make (Array.length needed_libs) MARK
  in

  let dynamic_frags =
    let dynamic_needed_frags = Array.make (Array.length needed_libs) MARK in
      for i = 0 to (Array.length needed_libs) - 1 do
        let fixup =
          new_fixup ("needed library name fixup: " ^ needed_libs.(i))
        in
          dynamic_needed_frags.(i) <-
            elf32_dyn_frag (DT_NEEDED, SUB (M_POS fixup,
                                            M_POS dynstr_section_fixup));
          dynamic_needed_strtab_frags.(i) <-
            DEF (fixup, ZSTRING needed_libs.(i))
      done;
      (SEQ [|
         SEQ dynamic_needed_frags;
         elf32_dyn_frag (DT_STRTAB, M_POS dynstr_section_fixup);
         elf32_dyn_frag (DT_STRSZ, M_SZ dynstr_section_fixup);

         elf32_dyn_frag (DT_SYMTAB, M_POS dynsym_section_fixup);
         elf32_dyn_frag (DT_SYMENT, IMM elf32_symsize);

         elf32_dyn_frag (DT_HASH, M_POS hash_section_fixup);
         elf32_dyn_frag (DT_PLTGOT, M_POS got_plt_section_fixup);

         elf32_dyn_frag (DT_PLTREL, IMM (elf32_num_of_dyn_tag DT_RELA));
         elf32_dyn_frag (DT_PLTRELSZ, M_SZ rela_plt_section_fixup);
         elf32_dyn_frag (DT_JMPREL, M_POS rela_plt_section_fixup);

         elf32_dyn_frag (DT_NULL, IMM 0L)
       |])
  in

  let null_strtab_fixup = new_fixup "null dynstrtab entry" in
  let null_strtab_frag = DEF (null_strtab_fixup, ZSTRING "") in
  let null_symtab_frag = (symbol
                            ~string_table_fixup: dynstr_section_fixup
                            ~name_string_fixup: null_strtab_fixup
                            ~sym_target_fixup: None
                            ~st_bind: STB_LOCAL
                            ~st_type: STT_NOTYPE
                            ~st_shndx: 0L) in

  let dynsym_frags = (null_symtab_frag ::
                        (require_symtab_frags @
                           global_text_symtab_frags @
                           local_text_symtab_frags @
                           rodata_symtab_frags @
                           data_symtab_frags @
                           bss_symtab_frags))
  in

  let dynstr_frags = (null_strtab_frag ::
                        (require_strtab_frags @
                           global_text_strtab_frags @
                           local_text_strtab_frags @
                           rodata_strtab_frags @
                           data_strtab_frags @
                           bss_strtab_frags @
                           (Array.to_list dynamic_needed_strtab_frags)))
  in

  let interp_section =

    DEF (interp_section_fixup, ZSTRING
           (if sess.Session.sess_targ = FreeBSD_x86_elf
            then "/libexec/ld-elf.so.1"
            else "/lib/ld-linux.so.2"))
  in

  let text_section =
    DEF (text_section_fixup,
         SEQ (Array.of_list text_body_frags))
  in
  let rodata_section =
    DEF (rodata_section_fixup,
         SEQ (Array.of_list rodata_body_frags))
  in
  let data_section =
    DEF (data_section_fixup,
         SEQ (Array.of_list data_body_frags))
  in
  let bss_section =
    DEF (bss_section_fixup,
         SEQ (Array.of_list bss_body_frags))
  in
  let dynsym_section =
    DEF (dynsym_section_fixup,
         SEQ (Array.of_list dynsym_frags))
  in
  let dynstr_section =
    DEF (dynstr_section_fixup,
         SEQ (Array.of_list dynstr_frags))
  in

  let hash_section =
    let n_syms = !n_syms in

    DEF (hash_section_fixup,
         (* Worst hashtable ever: one chain. *)
         SEQ [|
           WORD (TY_u32, IMM 1L);          (* nbucket *)
           WORD (TY_u32,                   (* nchain *)
                 IMM (Int64.of_int n_syms));
           WORD (TY_u32, IMM 1L);          (* bucket 0 => symbol 1. *)
           SEQ
             begin
               Array.init
                 n_syms
                 (fun i ->
                    let next = (* chain[i] => if last then 0 else i+1 *)
                      if i > 0 && i < (n_syms-1)
                      then Int64.of_int (i+1)
                      else 0L
                    in
                      WORD (TY_u32, IMM next))
             end;
         |])
  in

  let plt_section =
    DEF (plt_section_fixup,
         SEQ (Array.of_list plt_frags))
  in

  let got_plt_section =
    DEF (got_plt_section_fixup,
         SEQ (Array.of_list got_plt_frags))
  in

  let rela_plt_section =
    DEF (rela_plt_section_fixup,
         SEQ (Array.of_list rela_plt_frags))
  in

  let dynamic_section =
    DEF (dynamic_section_fixup, dynamic_frags)
  in

  let note_rust_section =
    DEF (note_rust_section_fixup,
         (Asm.note_rust_frags crate.node.Ast.crate_meta))
  in


  let page_alignment = 0x1000 in

  let align_both i =
    ALIGN_FILE (page_alignment,
                (ALIGN_MEM (page_alignment, i)))
  in

  let def_aligned f i =
    align_both
      (SEQ [| DEF(f,i);
              (align_both MARK)|])
  in

  let debug_aranges_section =
    def_aligned
      sem.Semant.ctxt_debug_aranges_fixup
      dwarf.Dwarf.debug_aranges
  in
  let debug_pubnames_section =
    def_aligned
      sem.Semant.ctxt_debug_pubnames_fixup
      dwarf.Dwarf.debug_pubnames
  in
  let debug_info_section =
    def_aligned
      sem.Semant.ctxt_debug_info_fixup
      dwarf.Dwarf.debug_info
  in
  let debug_abbrev_section =
    def_aligned
      sem.Semant.ctxt_debug_abbrev_fixup
      dwarf.Dwarf.debug_abbrev
  in
  let debug_line_section =
    def_aligned
      sem.Semant.ctxt_debug_line_fixup
      dwarf.Dwarf.debug_line
  in
  let debug_frame_section =
    def_aligned sem.Semant.ctxt_debug_frame_fixup dwarf.Dwarf.debug_frame
  in

  let load_address = 0x0804_8000L in

    SEQ
      [|
        MEMPOS load_address;
        ALIGN_FILE
          (segment_2_align,
           DEF
             (segment_2_fixup,
              SEQ
                [|
                  DEF (sem.Semant.ctxt_image_base_fixup, MARK);
                  elf_header;
                  ALIGN_FILE
                    (segment_0_align,
                     DEF
                       (segment_0_fixup,
                        SEQ
                          [|
                            DEF (program_header_table_fixup,
                                 program_header_table);
                          |]));
                  ALIGN_FILE
                    (segment_1_align,
                     DEF (segment_1_fixup, interp_section));
                  text_section;
                  rodata_section;
                  dynsym_section;
                  dynstr_section;
                  hash_section;
                  plt_section;
                  rela_plt_section;
                  debug_aranges_section;
                  debug_pubnames_section;
                  debug_info_section;
                  debug_abbrev_section;
                  debug_line_section;
                  debug_frame_section;
                |]));
        ALIGN_FILE
          (segment_3_align,
           DEF
             (segment_3_fixup,
              SEQ
                [|
                  data_section;
                  got_plt_section;
                  bss_section;
                  ALIGN_FILE
                    (segment_4_align,
                     DEF (segment_4_fixup,
                          dynamic_section));
                  ALIGN_FILE
                    (segment_5_align,
                     DEF (segment_5_fixup,
                          note_rust_section));
                |]));
        DEF (shstrtab_section_fixup,
             shstrtab_section);
        DEF (section_header_table_fixup,
             section_header_table);
      |]
;;

let emit_file
    (sess:Session.sess)
    (crate:Ast.crate)
    (code:Asm.frag)
    (data:Asm.frag)
    (sem:Semant.ctxt)
    (dwarf:Dwarf.debug_records)
    : unit =

  let text_frags = Hashtbl.create 4 in
  let rodata_frags = Hashtbl.create 4 in
  let data_frags = Hashtbl.create 4 in
  let bss_frags = Hashtbl.create 4 in
  let required_fixups = Hashtbl.create 4 in

  (*
   * Startup on elf-linux is more complex than in win32. It's
   * thankfully documented in some detail around the net.
   *
   *   - The elf entry address is for _start.
   *
   *   - _start pushes:
   *
   *       eax   (should be zero)
   *       esp   (holding the kernel-provided stack end)
   *       edx   (address of _rtld_fini)
   *       address of _fini
   *       address of _init
   *       ecx   (argv)
   *       esi   (argc)
   *       address of main
   *
   *     and then calls __libc_start_main@plt.
   *
   *   - This means any sensible binary has a PLT. Fun. So
   *     We call into the PLT, which itself is just a bunch
   *     of indirect jumps through slots in the GOT, and wind
   *     up in __libc_start_main. Which calls _init, then
   *     essentially exit(main(argc,argv)).
   *)


  let init_fixup = new_fixup "_init function entry" in
  let fini_fixup = new_fixup "_fini function entry" in
  let (start_fixup, rust_start_fixup) =
    if sess.Session.sess_library_mode
    then (None, None)
    else (Some (new_fixup "start function entry"),
          Some (Semant.require_native sem REQUIRED_LIB_rustrt "rust_start"))
  in
  let libc_start_main_fixup = new_fixup "__libc_start_main@plt stub" in

  let start_fn _ =
    let start_fixup =
      match start_fixup with
          None -> bug () "missing start fixup in non-library mode"
        | Some s -> s
    in
    let e = X86.new_emitter_without_vregs () in
    let push_r32 r = Il.emit e
      (Il.Push (Il.Cell (Il.Reg (Il.Hreg r, Il.ValTy Il.Bits32))))
    in
    let push_pos32 = X86.push_pos32 e in

      Il.emit e (Il.unary Il.UMOV (X86.rc X86.ebp) (X86.immi 0L));
      Il.emit e (Il.Pop (X86.rc X86.esi));
      Il.emit e (Il.unary Il.UMOV (X86.rc X86.ecx) (X86.ro X86.esp));
      Il.emit e (Il.binary Il.AND
                   (X86.rc X86.esp) (X86.ro X86.esp)
                   (X86.immi 0xfffffffffffffff0L));

      push_r32 X86.eax;
      push_r32 X86.esp;
      push_r32 X86.edx;
      push_pos32 fini_fixup;
      push_pos32 init_fixup;
      push_r32 X86.ecx;
      push_r32 X86.esi;
      push_pos32 start_fixup;
      Il.emit e (Il.call
                   (Il.Reg (Il.Hreg X86.eax, Il.ValTy Il.Bits32))
                   (Il.direct_code_ptr libc_start_main_fixup));
      X86.frags_of_emitted_quads sess e
  in

  let do_nothing_fn _ =
    let e = X86.new_emitter_without_vregs () in
      Il.emit e Il.Ret;
      X86.frags_of_emitted_quads sess e
  in

  let main_fn _ =
    match (start_fixup, rust_start_fixup, sem.Semant.ctxt_main_fn_fixup) with
        (None, _, _)
      | (_, None, _)
      | (_, _, None) -> MARK
      | (Some start_fixup,
         Some rust_start_fixup,
         Some main_fn_fixup) ->
          let e = X86.new_emitter_without_vregs () in
            X86.objfile_start e
              ~start_fixup
              ~rust_start_fixup
              ~main_fn_fixup
              ~crate_fixup: sem.Semant.ctxt_crate_fixup
              ~indirect_start: false;
            X86.frags_of_emitted_quads sess e
  in

  let needed_libs =
    [|
      if sess.Session.sess_targ = FreeBSD_x86_elf
      then "libc.so.7"
      else "libc.so.6";
      "librustrt.so"
    |]
  in

  let _ =
    if not sess.Session.sess_library_mode
    then
      begin
        htab_put text_frags (Some "_start") (start_fn());
        htab_put text_frags (Some "_init")
          (DEF (init_fixup, do_nothing_fn()));
        htab_put text_frags (Some "_fini")
          (DEF (fini_fixup, do_nothing_fn()));
        htab_put text_frags (Some "main") (main_fn ());
        htab_put required_fixups "__libc_start_main" libc_start_main_fixup;
      end;
    htab_put text_frags None code;
    htab_put rodata_frags None data;

    if sess.Session.sess_targ = FreeBSD_x86_elf
    then
      (* 
       * FreeBSD wants some extra symbols in .bss so its libc can fill
       * them in, I think.
       *)
      List.iter
        (fun x -> htab_put bss_frags (Some x) (WORD (TY_u32, (IMM 0L))))
        [
          "environ";
          "optind";
          "optarg";
          "_CurrentRuneLocale";
          "__stack_chk_guard";
          "__mb_sb_limit";
          "__isthreaded";
          "__stdinp";
          "__stderrp";
          "__stdoutp";
        ];

    Hashtbl.iter
      begin
        fun _ tab ->
          Hashtbl.iter
            begin
              fun name fixup ->
                htab_put required_fixups name fixup
            end
            tab
      end
      sem.Semant.ctxt_native_required
  in

  let all_frags =
    elf32_linux_x86_file
      ~sess
      ~crate
      ~entry_name: "_start"
      ~text_frags
      ~data_frags
      ~bss_frags
      ~dwarf
      ~sem
      ~rodata_frags
      ~required_fixups
      ~needed_libs
  in
    write_out_frag sess true all_frags
;;

let elf_magic = "\x7fELF";;

let sniff
    (sess:Session.sess)
    (filename:filename)
    : asm_reader option =
  try
    let stat = Unix.stat filename in
      if (stat.Unix.st_kind = Unix.S_REG) &&
        (stat.Unix.st_size > 4)
      then
        let ar = new_asm_reader sess filename in
        let _ = log sess "sniffing ELF file" in
          if (ar.asm_get_zstr_padded 4) = elf_magic
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
  let sects = Hashtbl.create 0 in
  let _ = log sess "reading sections" in
  let elf_id = ar.asm_get_zstr_padded 4 in
  let _ = assert (elf_id = elf_magic) in

  let _ = ar.asm_seek 0x10 in
  let _ = ar.asm_adv_u16 () in (* e_type *)
  let _ = ar.asm_adv_u16 () in (* e_machine *)
  let _ = ar.asm_adv_u32 () in (* e_version *)
  let _ = ar.asm_adv_u32 () in (* e_entry *)
  let _ = ar.asm_adv_u32 () in (* e_phoff *)
  let e_shoff = ar.asm_get_u32 () in (* e_shoff *)
  let _ = ar.asm_adv_u32 () in (* e_flags *)
  let _ = ar.asm_adv_u16 () in (* e_ehsize *)
  let _ = ar.asm_adv_u16 () in (* e_phentsize *)
  let _ = ar.asm_adv_u16 () in (* e_phnum *)
  let e_shentsize = ar.asm_get_u16 () in
  let e_shnum = ar.asm_get_u16 () in
  let e_shstrndx = ar.asm_get_u16 () in
  let _ = log sess
    "%d ELF section headers, %d bytes each, starting at 0x%x"
    e_shnum e_shentsize e_shoff
  in
  let _ = log sess "section %d is .shstrtab" e_shstrndx in

  let read_section_hdr n =
    let _ = ar.asm_seek (e_shoff + n * e_shentsize) in
    let str_off = ar.asm_get_u32() in
    let _ = ar.asm_adv_u32() in (* sh_type  *)
    let _ = ar.asm_adv_u32() in (* sh_flags *)
    let _ = ar.asm_adv_u32() in (* sh_addr *)
    let off = ar.asm_get_u32() in (* sh_off *)
    let size = ar.asm_get_u32() in (* sh_size *)
    let _ = ar.asm_adv_u32() in (* sh_link *)
    let _ = ar.asm_adv_u32() in (* sh_info *)
    let _ = ar.asm_adv_u32() in (* sh_addralign *)
    let _ = ar.asm_adv_u32() in (* sh_entsize *)
      (str_off, off, size)
  in

  let (_, str_base, _) = read_section_hdr e_shstrndx in

  let _ = ar.asm_seek e_shoff in
    for i = 0 to (e_shnum - 1) do
      let (str_off, off, size) = read_section_hdr i in
      let _ = ar.asm_seek (str_base + str_off) in
      let name = ar.asm_get_zstr() in
        log sess "section %d: %s, size %d, offset 0x%x" i name size off;
        Hashtbl.add sects name (off, size);
    done;
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
