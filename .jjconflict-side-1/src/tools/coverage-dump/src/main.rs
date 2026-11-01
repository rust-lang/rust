mod covfun;
mod covmap;
mod llvm_utils;
mod parser;
mod prf_names;

fn main() -> anyhow::Result<()> {
    use anyhow::Context as _;

    let args = std::env::args().collect::<Vec<_>>();

    // The coverage-dump tool already needs `rustc_demangle` in order to read
    // coverage metadata, so it's very easy to also have a separate mode that
    // turns it into a command-line demangler for use by coverage-run tests.
    if &args[1..] == &["--demangle"] {
        return demangle();
    }

    let llvm_ir_path = args.get(1).context("LLVM IR file not specified")?;
    let llvm_ir = std::fs::read_to_string(llvm_ir_path).context("couldn't read LLVM IR file")?;

    let filename_tables = covmap::make_filename_tables(&llvm_ir)?;
    let function_names = crate::prf_names::make_function_names_table(&llvm_ir)?;
    crate::covfun::dump_covfun_mappings(&llvm_ir, &filename_tables, &function_names)?;

    Ok(())
}

fn demangle() -> anyhow::Result<()> {
    use std::fmt::Write as _;

    let stdin = std::io::read_to_string(std::io::stdin())?;
    let mut output = String::with_capacity(stdin.len());
    for line in stdin.lines() {
        writeln!(output, "{:#}", rustc_demangle::demangle(line))?;
    }
    print!("{output}");
    Ok(())
}
