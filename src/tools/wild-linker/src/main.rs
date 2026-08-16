fn main() {
    if let Err(error) = run() {
        libwild::error::report_error_and_exit(&error)
    }
}

fn run() -> libwild::error::Result {
    let mut args = libwild::Args::new(|| std::env::args())?;
    args.parse(|| std::env::args())?;

    if libwild::should_fork(&args) {
        // Safety: We haven't spawned any threads yet.
        unsafe { libwild::run_in_subprocess(args) };
    } else {
        // Run the linker in this process without forking.
        libwild::setup_tracing(&args)?;
        libwild::run(args)
    }
}
