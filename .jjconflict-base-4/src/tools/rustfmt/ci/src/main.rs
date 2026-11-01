mod build_and_test;
mod clippy_config_check;
mod common;
mod integration;

fn main() {
    let mut args = std::env::args().skip(1);
    if let Err(error) = match args.next().as_deref() {
        Some("integration") => integration::runner(&mut args),
        Some("build-and-test") => build_and_test::runner(),
        Some("clippy-config-check") => clippy_config_check::runner(),
        Some(arg) => Err(format!(
            "Expected `integration`, `build-and-test`, or `clippy-config-check` \
                as first argument, found {arg:?}"
        )),
        None => Err(
            "Expected `integration`, `build-and-test`, or `clippy-config-check` \
                as first argument, found nothing"
                .to_string(),
        ),
    } {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
