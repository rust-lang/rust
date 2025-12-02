//@ ignore-cross-compile
// gnu ld is confused with intermediate files having multibytes characters in their names:
// = note: ld.exe: cannot find f0d5ff18d6510ebc-???_???_??????????_?_?????_?_???????.d50c2 \
// 4c0c4ea93cc-cgu.0.rcgu.o: Invalid argument
// as this is not something rustc can fix by itself,
// we just skip the test on windows-gnu for now. Hence:
//@ ignore-windows-gnu

use run_make_support::{rfs, rustc};

// This test make sure we don't crash when lto creates output files with long names.
// cn characters can be multi-byte and thus trigger the long filename reduction code more easily.
// we need to make sure that the code is properly generating names at char boundaries.
// as reported in issue #147975
fn main() {
    let lto_flags = ["-Clto", "-Clto=yes", "-Clto=off", "-Clto=thin", "-Clto=fat"];
    for prefix_len in 0..4 {
        let prefix: String = std::iter::repeat("_").take(prefix_len).collect();
        let main_file = format!("{}ⵅⴻⵎⵎⴻⵎ_ⴷⵉⵎⴰ_ⵖⴻⴼ_ⵢⵉⵙⴻⴽⴽⵉⵍⴻⵏ_ⵏ_ⵡⴰⵟⴰⵙ_ⵏ_ⵢⵉⴱⵢⵜⴻⵏ.rs", prefix);
        rfs::write(&main_file, "fn main() {}\n");
        for flag in lto_flags {
            rustc().input(&main_file).arg(flag).run();
        }
    }
}
