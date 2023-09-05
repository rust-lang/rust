mod covfun;
mod parser;
mod prf_names;

fn main() -> anyhow::Result<()> {
    use anyhow::Context as _;

    let args = std::env::args().collect::<Vec<_>>();

    let llvm_ir_path = args.get(1).context("LLVM IR file not specified")?;
    let llvm_ir = std::fs::read_to_string(llvm_ir_path).context("couldn't read LLVM IR file")?;

    let function_names = crate::prf_names::make_function_names_table(&llvm_ir)?;
    crate::covfun::dump_covfun_mappings(&llvm_ir, &function_names)?;

    Ok(())
}
