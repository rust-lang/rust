use enzyme::Cli;

fn main() {
    let args = Cli::parse();
    match setup(args) {
        Ok(_) => (),
        Err(e) => panic!("failed due to: {}", e),
    }
}

fn setup(args: Cli) -> Result<(), String> {
    enzyme::download(args.clone())?;
    enzyme::build(args.clone())?;
    enzyme::generate_bindings(args)?;
    Ok(())
}
