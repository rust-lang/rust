fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "musl-reference-tests")]
    musl_reference_tests::generate();
}

#[cfg(feature = "musl-reference-tests")]
mod musl_reference_tests {
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::fs;
    use std::process::Command;

    // Number of tests to generate for each function
    const NTESTS: usize = 500;

    // These files are all internal functions or otherwise miscellaneous, not
    // defining a function we want to test.
    const IGNORED_FILES: &[&str] = &[
        "expo2.rs",
        "fenv.rs",
        "k_cos.rs",
        "k_cosf.rs",
        "k_expo2.rs",
        "k_expo2f.rs",
        "k_sin.rs",
        "k_sinf.rs",
        "k_tan.rs",
        "k_tanf.rs",
        "mod.rs",
        "rem_pio2.rs",
        "rem_pio2_large.rs",
        "rem_pio2f.rs",
    ];

    struct Function {
        name: String,
        args: Vec<Ty>,
        ret: Ty,
        tests: Vec<Test>,
    }

    enum Ty {
        F32,
        F64,
        I32,
        Bool,
    }

    struct Test {
        inputs: Vec<i64>,
        output: i64,
    }

    pub fn generate() {
        let files = fs::read_dir("src/math")
            .unwrap()
            .map(|f| f.unwrap().path())
            .collect::<Vec<_>>();

        let mut math = Vec::new();
        for file in files {
            if IGNORED_FILES.iter().any(|f| file.ends_with(f)) {
                continue;
            }

            println!("generating musl reference tests in {:?}", file);

            let contents = fs::read_to_string(file).unwrap();
            let mut functions = contents.lines().filter(|f| f.starts_with("pub fn"));
            let function_to_test = functions.next().unwrap();
            if functions.next().is_some() {
                panic!("more than one function in");
            }

            math.push(parse(function_to_test));
        }

        // Generate a bunch of random inputs for each function. This will
        // attempt to generate a good set of uniform test cases for exercising
        // all the various functionality.
        generate_random_tests(&mut math, &mut rand::thread_rng());

        // After we have all our inputs, use the x86_64-unknown-linux-musl
        // target to generate the expected output.
        generate_test_outputs(&mut math);

        // ... and now that we have both inputs and expected outputs, do a bunch
        // of codegen to create the unit tests which we'll actually execute.
        generate_unit_tests(&math);
    }

    /// A "poor man's" parser for the signature of a function
    fn parse(s: &str) -> Function {
        let s = eat(s, "pub fn ");
        let pos = s.find('(').unwrap();
        let name = &s[..pos];
        let s = &s[pos + 1..];
        let end = s.find(')').unwrap();
        let args = s[..end]
            .split(',')
            .map(|arg| {
                let colon = arg.find(':').unwrap();
                parse_ty(arg[colon + 1..].trim())
            })
            .collect::<Vec<_>>();
        let tail = &s[end + 1..];
        let tail = eat(tail, " -> ");
        let ret = parse_ty(tail.trim().split(' ').next().unwrap());

        return Function {
            name: name.to_string(),
            args,
            ret,
            tests: Vec::new(),
        };

        fn parse_ty(s: &str) -> Ty {
            match s {
                "f32" => Ty::F32,
                "f64" => Ty::F64,
                "i32" => Ty::I32,
                "bool" => Ty::Bool,
                other => panic!("unknown type `{}`", other),
            }
        }

        fn eat<'a>(s: &'a str, prefix: &str) -> &'a str {
            if s.starts_with(prefix) {
                &s[prefix.len()..]
            } else {
                panic!("{:?} didn't start with {:?}", s, prefix)
            }
        }
    }

    fn generate_random_tests<R: Rng>(functions: &mut [Function], rng: &mut R) {
        for function in functions {
            for _ in 0..NTESTS {
                function.tests.push(generate_test(&function.args, rng));
            }
        }

        fn generate_test<R: Rng>(args: &[Ty], rng: &mut R) -> Test {
            let inputs = args.iter().map(|ty| ty.gen_i64(rng)).collect();
            // zero output for now since we'll generate it later
            Test { inputs, output: 0 }
        }
    }

    impl Ty {
        fn gen_i64<R: Rng>(&self, r: &mut R) -> i64 {
            match self {
                Ty::F32 => r.gen::<f32>().to_bits().into(),
                Ty::F64 => r.gen::<f64>().to_bits() as i64,
                Ty::I32 => {
                    if r.gen_range(0, 10) < 1 {
                        let i = *[i32::max_value(), 0, i32::min_value()].choose(r).unwrap();
                        i.into()
                    } else {
                        r.gen::<i32>().into()
                    }
                }
                Ty::Bool => r.gen::<bool>() as i64,
            }
        }

        fn libc_ty(&self) -> &'static str {
            match self {
                Ty::F32 => "f32",
                Ty::F64 => "f64",
                Ty::I32 => "i32",
                Ty::Bool => "i32",
            }
        }
    }

    fn generate_test_outputs(functions: &mut [Function]) {
        let mut src = String::new();
        let dst = std::env::var("OUT_DIR").unwrap();

        // Generate a program which will run all tests with all inputs in
        // `functions`. This program will write all outputs to stdout (in a
        // binary format).
        src.push_str("use std::io::Write;");
        src.push_str("fn main() {");
        src.push_str("let mut result = Vec::new();");
        for function in functions.iter_mut() {
            src.push_str("unsafe {");
            src.push_str("extern { fn ");
            src.push_str(&function.name);
            src.push_str("(");
            for (i, arg) in function.args.iter().enumerate() {
                src.push_str(&format!("arg{}: {},", i, arg.libc_ty()));
            }
            src.push_str(") -> ");
            src.push_str(function.ret.libc_ty());
            src.push_str("; }");

            src.push_str(&format!("static TESTS: &[[i64; {}]]", function.args.len()));
            src.push_str(" = &[");
            for test in function.tests.iter() {
                src.push_str("[");
                for val in test.inputs.iter() {
                    src.push_str(&val.to_string());
                    src.push_str(",");
                }
                src.push_str("],");
            }
            src.push_str("];");

            src.push_str("for test in TESTS {");
            src.push_str("let output = ");
            src.push_str(&function.name);
            src.push_str("(");
            for (i, arg) in function.args.iter().enumerate() {
                src.push_str(&match arg {
                    Ty::F32 => format!("f32::from_bits(test[{}] as u32)", i),
                    Ty::F64 => format!("f64::from_bits(test[{}] as u64)", i),
                    Ty::I32 => format!("test[{}] as i32", i),
                    Ty::Bool => format!("test[{}] as i32", i),
                });
                src.push_str(",");
            }
            src.push_str(");");
            src.push_str("let output = ");
            src.push_str(match function.ret {
                Ty::F32 => "output.to_bits() as i64",
                Ty::F64 => "output.to_bits() as i64",
                Ty::I32 => "output as i64",
                Ty::Bool => "output as i64",
            });
            src.push_str(";");
            src.push_str("result.extend_from_slice(&output.to_le_bytes());");

            src.push_str("}");

            src.push_str("}");
        }

        src.push_str("std::io::stdout().write_all(&result).unwrap();");

        src.push_str("}");

        let path = format!("{}/gen.rs", dst);
        fs::write(&path, src).unwrap();

        // Make it somewhat pretty if something goes wrong
        drop(Command::new("rustfmt").arg(&path).status());

        // Compile and execute this tests for the musl target, assuming we're an
        // x86_64 host effectively.
        let status = Command::new("rustc")
            .current_dir(&dst)
            .arg(&path)
            .arg("--target=x86_64-unknown-linux-musl")
            .status()
            .unwrap();
        assert!(status.success());
        let output = Command::new("./gen")
            .current_dir(&dst)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());

        // Map all the output bytes back to an `i64` and then shove it all into
        // the expected results.
        let mut results =
            output.stdout.chunks_exact(8)
             .map(|buf| {
                 let mut exact = [0; 8];
                 exact.copy_from_slice(buf);
                 i64::from_le_bytes(exact)
             });

        for test in functions.iter_mut().flat_map(|f| f.tests.iter_mut()) {
            test.output = results.next().unwrap();
        }
        assert!(results.next().is_none());
    }

    /// Codegens a file which has a ton of `#[test]` annotations for all the
    /// tests that we generated above.
    fn generate_unit_tests(functions: &[Function]) {
        let mut src = String::new();
        let dst = std::env::var("OUT_DIR").unwrap();

        for function in functions {
            src.push_str("#[test]");
            src.push_str("fn ");
            src.push_str(&function.name);
            src.push_str("_matches_musl() {");
            src.push_str(&format!("static TESTS: &[([i64; {}], i64)]", function.args.len()));
            src.push_str(" = &[");
            for test in function.tests.iter() {
                src.push_str("([");
                for val in test.inputs.iter() {
                    src.push_str(&val.to_string());
                    src.push_str(",");
                }
                src.push_str("],");
                src.push_str(&test.output.to_string());
                src.push_str("),");
            }
            src.push_str("];");

            src.push_str("for (test, expected) in TESTS {");
            src.push_str("let output = ");
            src.push_str(&function.name);
            src.push_str("(");
            for (i, arg) in function.args.iter().enumerate() {
                src.push_str(&match arg {
                    Ty::F32 => format!("f32::from_bits(test[{}] as u32)", i),
                    Ty::F64 => format!("f64::from_bits(test[{}] as u64)", i),
                    Ty::I32 => format!("test[{}] as i32", i),
                    Ty::Bool => format!("test[{}] as i32", i),
                });
                src.push_str(",");
            }
            src.push_str(");");
            src.push_str(match function.ret {
                Ty::F32 => "if _eqf(output, f32::from_bits(*expected as u32)).is_ok() { continue }",
                Ty::F64 => "if _eq(output, f64::from_bits(*expected as u64)).is_ok() { continue }",
                Ty::I32 => "if output as i64 == expected { continue }",
                Ty::Bool => unreachable!(),
            });

            src.push_str(r#"
                panic!("INPUT: {:?} EXPECTED: {:?} ACTUAL {:?}", test, expected, output);
            "#);
            src.push_str("}");

            src.push_str("}");
        }

        let path = format!("{}/tests.rs", dst);
        fs::write(&path, src).unwrap();

        // Try to make it somewhat pretty
        drop(Command::new("rustfmt").arg(&path).status());
    }
}
