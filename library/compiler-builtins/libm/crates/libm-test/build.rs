#[path = "../../configure.rs"]
mod configure;
use configure::Config;

fn main() {
    let cfg = Config::from_env();
    configure::emit_test_config(&cfg);
}
