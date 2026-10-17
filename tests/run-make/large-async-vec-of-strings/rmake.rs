// A regression test to ensure that an async function returning a large
// `Vec<String>` compiles within a reasonable time. The quadratic MIR
// blowup (#115327) manifseting itself earlier was caused by `StorageDead`
// on the unwind path preventing tail-sharing of drop chains in coroutines.
//
// The test generates a source file containing `vec!["string".to_string(), ...]`
// with 10,000 elements inside an async fn, then checks that it compiles.
// The timeout is set such that the test will warn about it running too long
// with the quadratic MIR growth. In the future time-outs could be converted
// into hard failures in the compiletest machinery.

//@ timeout: 10

use std::fs;

use run_make_support::rustc;

const ELEMENT_COUNT: usize = 10_000;

fn main() {
    let vec_elements = r#""string".to_string(),"#.repeat(ELEMENT_COUNT);

    let source = format!(
        r#"
            fn main() {{ produce_vector(); }}
            async fn produce_vector() -> Vec<String> {{
                vec![{vec_elements}]
            }}
        "#
    );

    fs::write("generated.rs", &source).unwrap();
    rustc().edition("2021").input("generated.rs").run();
}
