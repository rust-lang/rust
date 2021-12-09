fn main() {
    // Pull in jemalloc when enabled.
    #[cfg(feature = "jemalloc")]
    rustc_driver::set_jemalloc();

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
