use std::path::Path;

use run_make_support::{htmldocck, rfs, rustc, rustdoc};

pub fn scrape(extra_args: &[&str]) {
    let out_dir = Path::new("rustdoc");
    let crate_name = "foobar";
    let deps = rfs::read_dir("examples")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "rs"))
        .collect::<Vec<_>>();

    rustc().input("src/lib.rs").crate_name(crate_name).crate_type("lib").emit("metadata").run();

    let mut out_deps = Vec::with_capacity(deps.len());
    for dep in deps {
        let dep_stem = dep.file_stem().unwrap();
        let out_example = out_dir.join(format!("{}.calls", dep_stem.to_str().unwrap()));
        rustdoc()
            .input(&dep)
            .crate_name(&dep_stem)
            .crate_type("bin")
            .out_dir(&out_dir)
            .extern_(crate_name, format!("lib{crate_name}.rmeta"))
            .arg("-Zunstable-options")
            .arg("--scrape-examples-output-path")
            .arg(&out_example)
            .arg("--scrape-examples-target-crate")
            .arg(crate_name)
            .args(extra_args)
            .run();
        out_deps.push(out_example);
    }

    let mut rustdoc = rustdoc();
    rustdoc
        .input("src/lib.rs")
        .out_dir(&out_dir)
        .crate_name(crate_name)
        .crate_type("lib")
        .arg("-Zunstable-options");
    for dep in out_deps {
        rustdoc.arg("--with-examples").arg(dep);
    }
    rustdoc.run();

    htmldocck().arg(out_dir).arg("src/lib.rs").run();
}
