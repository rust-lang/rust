#[path = "../../configure.rs"]
mod configure;

fn main() {
    let cfg = configure::Config::from_env();
    configure::emit_libm_config(&cfg);
}
