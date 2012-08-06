import driver::session;
import session::sess_os_to_meta_os;
import metadata::loader::meta_section_name;

fn get_target_strs(target_os: session::os) -> target_strs::t {
    return {
        module_asm: ~"",

        meta_sect_name: meta_section_name(sess_os_to_meta_os(target_os)),

        data_layout: match target_os {
          session::os_macos => {
            ~"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"+
                ~"f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-"+
                ~"s0:64:64-f80:128:128-n8:16:32:64"
          }

          session::os_win32 => {
            // FIXME: Test this. Copied from linux (#2398)
            ~"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"+
                ~"f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-"+
                ~"s0:64:64-f80:128:128-n8:16:32:64-S128"
          }

          session::os_linux => {
            ~"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"+
                ~"f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-"+
                ~"s0:64:64-f80:128:128-n8:16:32:64-S128"
          }

          session::os_freebsd => {
            ~"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"+
                ~"f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-"+
                ~"s0:64:64-f80:128:128-n8:16:32:64-S128"
          }
        },

        target_triple: match target_os {
          session::os_macos => ~"x86_64-apple-darwin",
          session::os_win32 => ~"x86_64-pc-mingw32",
          session::os_linux => ~"x86_64-unknown-linux-gnu",
          session::os_freebsd => ~"x86_64-unknown-freebsd",
        },

        cc_args: ~[~"-m64"]
    };
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
