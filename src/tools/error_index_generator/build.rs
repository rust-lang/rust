use std::path::PathBuf;
use std::{env, fs};
use walkdir::WalkDir;

fn main() {
    // The src directory (we are in src/tools/error_index_generator)
    // Note that we could skip one of the .. but this ensures we at least loosely find the right
    // directory.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("error_codes.rs");

    let error_codes_path = "../../../src/librustc_error_codes/error_codes.rs";

    println!("cargo:rerun-if-changed={}", error_codes_path);
    let file = fs::read_to_string(error_codes_path)
        .unwrap()
        .replace(": include_str!(\"./error_codes/", ": include_str!(\"./");
    let contents = format!("(|| {{\n{}\n}})()", file);
    fs::write(&out_dir.join("all_error_codes.rs"), &contents).unwrap();

    // We copy the md files as well to the target directory.
    for entry in WalkDir::new("../../../src/librustc_error_codes/error_codes") {
        let entry = entry.unwrap();
        match entry.path().extension() {
            Some(s) if s == "md" => {}
            _ => continue,
        }
        println!("cargo:rerun-if-changed={}", entry.path().to_str().unwrap());
        let md_content = fs::read_to_string(entry.path()).unwrap();
        fs::write(&out_dir.join(entry.file_name()), &md_content).unwrap();
    }

    let mut all = String::new();
    all.push_str(
        r###"
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
"###,
    );
    all.push_str(r#"include!(concat!(env!("OUT_DIR"), "/all_error_codes.rs"));"#);
    all.push_str("\nlong_codes\n");
    all.push_str("}\n");

    fs::write(&dest, all).unwrap();
}
