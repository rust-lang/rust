fn main() {
    // generate sha256 files
    // this avoids having to perform hashing at runtime
    let files = &[
        "static/css/rustdoc.css",
        "static/css/noscript.css",
        "static/css/normalize.css",
        "static/js/main.js",
        "static/js/search.js",
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
        "static/fonts/FiraSans-Regular.woff2",
        "static/fonts/FiraSans-Medium.woff2",
        "static/fonts/FiraSans-LICENSE.txt",
        "static/fonts/SourceSerif4-Regular.ttf.woff2",
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
        let bytes = std::fs::read(inpath).expect("static path exists");
        use sha2::Digest;
        let bytes = sha2::Sha256::digest(bytes);
        let mut digest = format!("-{bytes:x}");
        digest.truncate(9);
        let outpath = std::path::PathBuf::from(format!("{out_dir}/{path}.sha256"));
        std::fs::create_dir_all(outpath.parent().expect("all file paths are in a directory"))
            .expect("should be able to write to out_dir");
        std::fs::write(&outpath, digest.as_bytes()).expect("write to out_dir");
    }
}
