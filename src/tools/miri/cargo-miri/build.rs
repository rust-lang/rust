use vergen::vergen;

fn main() {
    // Don't rebuild miri when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // vergen
    let mut gen_config = vergen::Config::default();
    *gen_config.git_mut().sha_kind_mut() = vergen::ShaKind::Short;
    *gen_config.git_mut().commit_timestamp_kind_mut() = vergen::TimestampKind::DateOnly;
    vergen(gen_config).ok(); // Ignore failure (in case we are built outside a git repo)
}
