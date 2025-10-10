use std::str;

use sha2::Digest;
fn main() {
    // generate sha256 files
    // this avoids having to perform hashing at runtime
    let files = &[
        "static/css/rustdoc.css",
        "static/css/noscript.css",
        "static/css/normalize.css",
        "static/js/main.js",
        "static/js/search.js",
        "static/js/stringdex.js",
        "static/js/settings.js",
        "static/js/src-script.js",
        "static/js/storage.js",
        "static/js/scrape-examples.js",
        "static/COPYRIGHT.txt",
        "static/LICENSE-APACHE.txt",
        "static/LICENSE-MIT.txt",
        "static/images/rust-logo.svg",
        "static/images/favicon.svg",
        "static/images/favicon-32x32.png",
        "static/fonts/FiraSans-Italic.woff2",
        "static/fonts/FiraSans-Regular.woff2",
        "static/fonts/FiraSans-Medium.woff2",
        "static/fonts/FiraSans-MediumItalic.woff2",
        "static/fonts/FiraMono-Regular.woff2",
        "static/fonts/FiraMono-Medium.woff2",
        "static/fonts/FiraSans-LICENSE.txt",
        "static/fonts/SourceSerif4-Regular.ttf.woff2",
        "static/fonts/SourceSerif4-Semibold.ttf.woff2",
        "static/fonts/SourceSerif4-Bold.ttf.woff2",
        "static/fonts/SourceSerif4-It.ttf.woff2",
        "static/fonts/SourceSerif4-LICENSE.md",
        "static/fonts/SourceCodePro-Regular.ttf.woff2",
        "static/fonts/SourceCodePro-Semibold.ttf.woff2",
        "static/fonts/SourceCodePro-It.ttf.woff2",
        "static/fonts/SourceCodePro-LICENSE.txt",
        "static/fonts/NanumBarunGothic.ttf.woff2",
        "static/fonts/NanumBarunGothic-LICENSE.txt",
    ];
    let out_dir = std::env::var("OUT_DIR").expect("standard Cargo environment variable");
    for path in files {
        let inpath = format!("html/{path}");
        println!("cargo::rerun-if-changed={inpath}");
        let data_bytes = std::fs::read(&inpath).expect("static path exists");
        let hash_bytes = sha2::Sha256::digest(&data_bytes);
        let mut digest = format!("-{hash_bytes:x}");
        digest.truncate(9);
        let outpath = std::path::PathBuf::from(format!("{out_dir}/{path}.sha256"));
        std::fs::create_dir_all(outpath.parent().expect("all file paths are in a directory"))
            .expect("should be able to write to out_dir");
        std::fs::write(&outpath, digest.as_bytes()).expect("write to out_dir");
        let minified_path = std::path::PathBuf::from(format!("{out_dir}/{path}.min"));
        if path.ends_with(".js") || path.ends_with(".css") {
            let minified: String = if path.ends_with(".css") {
                minifier::css::minify(str::from_utf8(&data_bytes).unwrap())
                    .unwrap()
                    .to_string()
                    .into()
            } else {
                minifier::js::minify(str::from_utf8(&data_bytes).unwrap()).to_string().into()
            };
            std::fs::write(&minified_path, minified.as_bytes()).expect("write to out_dir");
        } else {
            std::fs::copy(&inpath, &minified_path).unwrap();
        }
    }
}
