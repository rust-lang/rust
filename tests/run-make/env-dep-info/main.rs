fn main() {
    env!("EXISTING_ENV");
    option_env!("EXISTING_OPT_ENV");
    option_env!("NONEXISTENT_OPT_ENV");
    option_env!("ESCAPE\nESCAPE\\");
}
