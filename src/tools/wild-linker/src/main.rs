fn main() {
    if let Err(error) = run() {
        libwild::error::report_error_and_exit(&error)
    }
}

fn run() -> libwild::error::Result {
    let args = libwild::Args::parse(|| std::env::args().skip(1))?;

    if args.should_fork() {
        // Safety: We haven't spawned any threads yet.
        unsafe { libwild::run_in_subprocess(args) };
    } else {
        // Run the linker in this process without forking.
        libwild::run(args)
    }
}
