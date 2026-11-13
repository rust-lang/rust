fn main() {
    // Simulate a localized (e.g. Japanese) `link.exe`, as printed when the
    // English language pack is not installed and `VSLANG=1033` has no effect.
    // This is "Creating library foo.dll.lib and object foo.dll.exp" in Japanese.
    println!("ライブラリ foo.dll.lib とオブジェクト foo.dll.exp を作成中");
    // A file name containing an `LNK####`-looking fragment must not be
    // mistaken for a diagnostic, which is why the matcher requires the
    // structured `LINK : warning LNK####:` form.
    println!("LNK2001.lib: progress message, not a diagnostic");
    for arg in std::env::args() {
        if arg == "run_make_lnk" {
            // Real diagnostics are structured as `LINK : warning LNK####:`.
            println!("LINK : warning LNK2001: unresolved external symbol foo");
            // The one code-bearing informational line has no `LINK : ` prefix
            // and keeps the exception that classifies it as `linker_info`.
            println!(
                "LNK6004: 'foo.exe' not found or not built by the last incremental link; \
                 performing full link"
            );
        }
    }
}
