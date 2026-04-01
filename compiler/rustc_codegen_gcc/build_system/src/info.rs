use crate::config::ConfigInfo;

pub fn run() -> Result<(), String> {
    let mut config = ConfigInfo::default();

    // We skip binary name and the `info` command.
    let mut args = std::env::args().skip(2);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("Display the path where the libgccjit will be located");
            return Ok(());
        }
        config.parse_argument(&arg, &mut args)?;
    }
    config.no_download = true;
    config.setup_gcc_path()?;
    if let Some(gcc_path) = config.gcc_path {
        println!("{gcc_path}");
    }
    Ok(())
}
