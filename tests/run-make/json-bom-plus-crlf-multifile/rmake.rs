//@ reference: input.byte-order-mark
//@ reference: input.crlf
//@ ignore-cross-compile
// ignore-tidy-linelength

use run_make_support::{cwd, diff, rfs, rustc};

fn main() {
    let aux_content = "\u{FEFF}\
pub fn test() {\r\n\
\r\n\
    let s : String = 1;  // Error in the middle of line.\r\n\
\r\n\
    let s : String = 1\r\n\
    ;  // Error before the newline.\r\n\
\r\n\
    let s : String =\r\n\
1;  // Error after the newline.\r\n\
\r\n\
    let s : String = (\r\n\
    );  // Error spanning the newline.\r\n\
}\r\n";

    rfs::write(cwd().join("json-bom-plus-crlf-multifile-aux.rs"), aux_content);

    let aux_bytes = rfs::read(cwd().join("json-bom-plus-crlf-multifile-aux.rs"));
    assert!(aux_bytes.starts_with(b"\xEF\xBB\xBF"), "File must start with UTF-8 BOM");
    assert!(aux_bytes.windows(2).any(|w| w == b"\r\n"), "File must contain CRLF line endings");

    let main_content = "\u{FEFF}\
#[path = \"json-bom-plus-crlf-multifile-aux.rs\"]\r\n\
mod json_bom_plus_crlf_multifile_aux;\r\n\
\r\n\
fn main() {\r\n\
    json_bom_plus_crlf_multifile_aux::test();\r\n\
}\r\n";

    rfs::write(cwd().join("json-bom-plus-crlf-multifile.rs"), main_content);

    let output = rustc()
        .input(cwd().join("json-bom-plus-crlf-multifile.rs"))
        .json("diagnostic-short")
        .error_format("json")
        .ui_testing()
        .run_fail()
        .stderr_utf8();

    diff()
        .expected_file("json-bom-plus-crlf-multifile.stderr")
        .actual_text("stderr", &output)
        .normalize(r"\\n", "\n")
        .normalize(r"\\r\\n", "\n")
        .normalize(r#""line_start":\d+"#, r#""line_start":LL"#)
        .normalize(r#""line_end":\d+"#, r#""line_end":LL"#)
        .normalize(r#""column_start":\d+"#, r#""column_start":CC"#)
        .normalize(r#""column_end":\d+"#, r#""column_end":CC"#)
        .normalize(r#""byte_start":\d+"#, r#""byte_start":XXX"#)
        .normalize(r#""byte_end":\d+"#, r#""byte_end":XXX"#)
        .normalize(r#""highlight_start":\d+"#, r#""highlight_start":XX"#)
        .normalize(r#""highlight_end":\d+"#, r#""highlight_end":XX"#)
        .normalize(
            r#""file_name":"[^"]*json-bom-plus-crlf-multifile-aux\.rs""#,
            r#""file_name":"$$DIR/json-bom-plus-crlf-multifile-aux.rs""#,
        )
        .normalize(
            r#""rendered":"(?:[^"]*/)?json-bom-plus-crlf-multifile-aux\.rs:\d+:\d+:"#,
            r#""rendered":"$$DIR/json-bom-plus-crlf-multifile-aux.rs:LL:CC:"#,
        )
        .normalize(
            r#""rendered":"(?:[^"]*/)?json-bom-plus-crlf-multifile-aux\.rs:\d+:\d+ error"#,
            r#""rendered":"$$DIR/json-bom-plus-crlf-multifile-aux.rs:LL:CC error"#,
        )
        .normalize(
            r#""rendered":"[a-zA-Z]:\\(?:[^\\"]+\\)*json-bom-plus-crlf-multifile-aux\.rs:\d+:\d+:"#,
            r#""rendered":"$$DIR/json-bom-plus-crlf-multifile-aux.rs:LL:CC:"#,
        )
        .normalize(
            r#""rendered":"[^"]*?error"#,
            r#""rendered":"$$DIR/json-bom-plus-crlf-multifile-aux.rs:LL:CC: error"#,
        )
        .run();
}
