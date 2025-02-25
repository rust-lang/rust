mod build_shared {
    include!("../std/build_shared.rs");
}

fn main() {
    let cfg = build_shared::Config::from_env();
    build_shared::configure_f16_f128(&cfg);
}
