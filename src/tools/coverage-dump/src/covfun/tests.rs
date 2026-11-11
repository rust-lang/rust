use super::{CovfunLineData, parse_covfun_line};

/// Integers in LLVM IR are not inherently signed/unsigned, and the text format tends
/// to emit them in signed form, so this helper function converts `i64` to `u64`.
fn as_u64(x: i64) -> u64 {
    x as u64
}

#[test]
fn parse_covfun_line_data() {
    struct Case {
        line: &'static str,
        expected: CovfunLineData,
    }
    let cases = &[
        // Copied from `trivial.ll`:
        Case {
            line: r#"@__covrec_49A9BAAE5F896E81u = linkonce_odr hidden constant <{ i64, i32, i64, i64, [9 x i8] }> <{ i64 5307978893922758273, i32 9, i64 445092354169400020, i64 6343436898695299756, [9 x i8] c"\01\01\00\01\01\03\01\00\0D" }>, section "__LLVM_COV,__llvm_covfun", align 8"#,
            expected: CovfunLineData {
                is_used: true,
                name_hash: as_u64(5307978893922758273),
                filenames_hash: as_u64(6343436898695299756),
                payload: b"\x01\x01\x00\x01\x01\x03\x01\x00\x0D".to_vec(),
            },
        },
        // Copied from `on-off-sandwich.ll`:
        Case {
            line: r#"@__covrec_D0CE53C5E64F319Au = linkonce_odr hidden constant <{ i64, i32, i64, i64, [14 x i8] }> <{ i64 -3400688559180533350, i32 14, i64 7307957714577672185, i64 892196767019953100, [14 x i8] c"\01\01\00\02\01\10\05\02\10\01\07\05\00\06" }>, section "__LLVM_COV,__llvm_covfun", align 8"#,
            expected: CovfunLineData {
                is_used: true,
                name_hash: as_u64(-3400688559180533350),
                filenames_hash: as_u64(892196767019953100),
                payload: b"\x01\x01\x00\x02\x01\x10\x05\x02\x10\x01\x07\x05\x00\x06".to_vec(),
            },
        },
        // Copied from `no-core.ll`:
        Case {
            line: r#"@__covrec_F8016FC82D46106u = linkonce_odr hidden constant <{ i64, i32, i64, i64, [9 x i8] }> <{ i64 1116917981370409222, i32 9, i64 -8857254680411629915, i64 -3625186110715410276, [9 x i8] c"\01\01\00\01\01\0C\01\00\0D" }>, section "__LLVM_COV,__llvm_covfun", align 8"#,
            expected: CovfunLineData {
                is_used: true,
                name_hash: as_u64(1116917981370409222),
                filenames_hash: as_u64(-3625186110715410276),
                payload: b"\x01\x01\x00\x01\x01\x0C\x01\x00\x0D".to_vec(),
            },
        },
    ];

    for &Case { line, ref expected } in cases {
        println!("- {line}");
        let line_data = parse_covfun_line(line).map_err(|e| e.to_string());
        assert_eq!(line_data.as_ref(), Ok(expected));
    }
}
