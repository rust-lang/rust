use std::collections::HashMap;
use std::io::BufWriter;
use std::{env, fs::File};
use std::path::Path;
use tidy::features::{collect_lang_features, collect_lib_features, Feature};

#[derive(serde::Serialize)]
struct FeaturesStatus {
    lang_features_status: HashMap<String, Feature>,
    lib_features_status: HashMap<String, Feature>,
}

fn main() {
    let library_path_str = env::args_os().nth(1).expect("library/ path required");
    let compiler_path_str = env::args_os().nth(2).expect("compiler/ path required");
    let output_path_str = env::args_os().nth(3).expect("output path required");
    let library_path = Path::new(&library_path_str);
    let compiler_path = Path::new(&compiler_path_str);
    let output_path = Path::new(&output_path_str);
    let lang_features_status = collect_lang_features(compiler_path, &mut false);
    let lib_features_status = collect_lib_features(library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features_status.contains_key(name))
        .collect();
    let features_status = FeaturesStatus {
        lang_features_status, lib_features_status
    };
    let writer = File::create(output_path).expect("output path should be a valid path");
    let writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(writer, &features_status).unwrap();
}
