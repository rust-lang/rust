use walkdir::WalkDir;
use std::path::PathBuf;
use std::{env, fs};

fn main() {
    // The src directory (we are in src/tools/error_index_generator)
    // Note that we could skip one of the .. but this ensures we at least loosely find the right
    // directory.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("error_codes.rs");
    let mut idx = 0;
    for entry in WalkDir::new("../../../src") {
        let entry = entry.unwrap();
        if entry.file_name() == "error_codes.rs" {
            println!("cargo:rerun-if-changed={}", entry.path().to_str().unwrap());
            let file = fs::read_to_string(entry.path()).unwrap()
                .replace("syntax::register_diagnostics!", "register_diagnostics!");
            let contents = format!("(|| {{\n{}\n}})();", file);

            fs::write(&out_dir.join(&format!("error_{}.rs", idx)), &contents).unwrap();

            idx += 1;
        }
    }

    let mut all = String::new();
    all.push_str(r###"
fn register_all() -> Vec<(&'static str, Option<&'static str>)> {
    let mut long_codes: Vec<(&'static str, Option<&'static str>)> = Vec::new();
    macro_rules! register_diagnostics {
        ($($ecode:ident: $message:expr,)*) => (
            register_diagnostics!{$($ecode:$message,)* ;}
        );

        ($($ecode:ident: $message:expr,)* ; $($code:ident,)*) => (
            $(
                {long_codes.extend([
                    (stringify!($ecode), Some($message)),
                ].iter());}
            )*
            $(
                {long_codes.extend([
                    stringify!($code),
                ].iter().cloned().map(|s| (s, None)).collect::<Vec<_>>());}
            )*
        )
    }
"###);
    for idx in 0..idx {
        all.push_str(&format!(r#"include!(concat!(env!("OUT_DIR"), "/error_{}.rs"));"#, idx));
        all.push_str("\n");
    }
    all.push_str("\nlong_codes\n");
    all.push_str("}\n");

    fs::write(&dest, all).unwrap();
}
